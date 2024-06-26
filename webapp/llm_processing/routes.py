from datetime import datetime
import tempfile
import zipfile
from . import llm_processing
from .. import socketio
from flask import render_template, current_app, flash, request, redirect, send_file, url_for
from .forms import LLMPipelineForm
import requests
import pandas as pd
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable, Optional
import os
from .read_strange_csv import read_and_save_csv
import secrets
from concurrent import futures
import io
from .utils import read_preprocessed_csv_from_zip, replace_personal_info, is_empty_string_nan_or_none
from io import BytesIO

server_connection: Optional[subprocess.Popen[Any]] = None
current_model = None

JobID = str
llm_jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)


new_model = False
llm_progress = {}

start_times = {}


def format_time(seconds):
    if seconds < 120:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}min"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    else:
        return f"{seconds / 86400:.1f}d"


def update_progress(job_id, progress: tuple[int, int, bool]):
    global llm_progress

    # Initialize llm_progress dictionary if not already initialized
    if 'llm_progress' not in globals():
        llm_progress = {}

    # Update progress dictionary
    llm_progress[job_id] = progress

    # Calculate elapsed time since the job started
    if job_id not in start_times:
        start_times[job_id] = time.time()
    elapsed_time = time.time() - start_times[job_id]

    # Calculate average time per progress step
    if progress[0] > 0:
        avg_time_per_step = elapsed_time / progress[0]
    else:
        avg_time_per_step = 0

    # Calculate estimated remaining time
    if progress[0] < progress[1]:
        remaining_steps = progress[1] - progress[0]
        estimated_remaining_time = avg_time_per_step * remaining_steps
    else:
        estimated_remaining_time = 0

    estimated_remaining_time = format_time(estimated_remaining_time)

    print("Progress: ", progress[0], " Total: ", progress[1],
          " Estimated Remaining Time: ", estimated_remaining_time)

    # Emit progress update via socketio
    socketio.emit('llm_progress_update', {
                  'job_id': job_id, 'progress': progress[0], 'total': progress[1], 'remaining_time': estimated_remaining_time})


def warning_job(job_id, message):
    global job_progress
    socketio.emit("progress_warning", {"job_id": job_id, "message": message})

@socketio.on('connect')
def handle_connect():
    print("Client Connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client Disconnected")


def extract_from_report(
        df: pd.DataFrame,
        model_name: str,
        prompt: str,
        symptoms: Iterable[str],
        temperature: float,
        grammar: str,
        model_path: str,
        server_path: str,
        ctx_size: int,
        n_gpu_layers: int,
        n_predict: int,
        job_id: int,
        zip_file_path: str,
        llamacpp_port: int,
        debug: bool = False,
        model_name_name: str = ""
) -> dict[Any]:
    print("Extracting from report")
    # Start server with correct model if not already running
    model_dir = Path(model_path)

    model_path = model_dir / model_name
    assert model_path.absolute().parent == model_dir

    global new_model
    global server_connection, current_model
    if current_model != model_name:
        server_connection and server_connection.kill()

        new_model = True
        server_connection = subprocess.Popen(
            [
                server_path,
                "--model",
                str(model_path),
                "--ctx-size",
                str(ctx_size),
                "--n-gpu-layers",
                str(n_gpu_layers),
                "--port",
                str(llamacpp_port),
                "-fa" # flash attention # use new llama cpp version
                # "--verbose",
            ],
        )
        current_model = model_name
        time.sleep(5)

    try:
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
    except KeyError:
        print("No proxy set")
        pass

    while True:
        try:
            response = requests.get(f"http://localhost:{llamacpp_port}/health")
            if response.json()["status"] == "ok":
                break
            elif response.json()["status"] == "error":
                socketio.emit("load_failed")
                return
            elif response.json()["status"] == "no slot available":
                warning_job(
                    job_id=job_id,
                    message="Model loaded, but currently no slots available",
                )
                break
            time.sleep(1)
        except requests.exceptions.ConnectionError:
            warning_job(
                job_id=job_id,
                message="Server connection error, will keep retrying ...",
            )
            time.sleep(5)

    print("Server running")

    new_model = False
    socketio.emit('load_complete')

    results = {}
    skipped = 0

    for i, (report, id) in enumerate(zip(df.report, df.id)):
        print("parsing report: ", i)
        if is_empty_string_nan_or_none(report):
            print("SKIPPING EMPTY REPORT!")
            skipped += 1
            update_progress(job_id=job_id, progress=(
                i + 1 - skipped, len(df) - skipped, True))
            continue
        for symptom in symptoms:

            prompt_formatted = prompt.format(symptom=symptom, report="".join(report))

            tokenized_result = requests.post(
                url=f"http://localhost:{llamacpp_port}/tokenize",
                json={"content": prompt_formatted},
            )

            num_prompt_tokens = len(tokenized_result.json()["tokens"])

            if num_prompt_tokens >= ctx_size - n_predict:
                print(
                    f"PROMPT MIGHT BE TOO LONG. PROMPT: {num_prompt_tokens} Tokens. CONTEXT SIZE: {ctx_size} Tokens. N-PREDICT: {n_predict} Tokens."
                )
                warning_job(
                    job_id=job_id,
                    message=f"Prompt might be too long. Prompt: {num_prompt_tokens} Tokens. Context size: {ctx_size} Tokens. N-Predict: {n_predict} Tokens.",
                )


            result = requests.post(
                url=f"http://localhost:{llamacpp_port}/completion",
                json={
                    "prompt": prompt.format(
                        symptom=symptom, report="".join(report)
                    ),
                    "n_predict": n_predict,
                    "temperature": temperature,
                    "grammar": grammar,
                },
                timeout=20 * 60,
            )

            summary = result.json()
            if id not in results:
                results[id] = {}

            results[id]['report'] = report
            results[id]['symptom'] = symptom
            results[id]['summary'] = summary

        print(f"Report {i} completed.")
        update_progress(job_id=job_id, progress=(
            i+1 - skipped, len(df) - skipped, True))

    socketio.emit('llm_progress_complete', {
                  'job_id': job_id, 'total_steps': len(df) - skipped})

    llm_metadata = {
        'model_name': model_name_name if model_name_name else model_name,
        'prompt': prompt,
        'symptoms': symptoms,
        'temperature': temperature,
        'n_predict': n_predict,
        'ctx_size': ctx_size,
        'grammar': grammar,
        'date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    return postprocess_grammar(results, df, llm_metadata, debug), zip_file_path


def postprocess_grammar(result, df, llm_metadata, debug=False):
    print("POSTPROCESSING GRAMMAR")

    extracted_data = []

    error_count = 0

    # Iterate over each report and its associated data
    for i, (id, info) in enumerate(result.items()):
        print(f"Processing report {i} of {len(result)}")
        # Get the first key in the dictionary (here assumed to be the relevant field)

        # Extract the content of the first field
        content = info['summary']['content']

        # Parse the content string into a dictionary
        try:
            if content.endswith('<|eot_id|>'):
                print("Remove eot_id")
                content = content[:-len('<|eot_id|>')]
            if content.endswith('</s>'):
                print("Remove </s>")
                content = content[:-len('</s>')]
            # search for last } in content and remove anything after that
            content = content[:content.rfind('}')+1]
            import ast

            # replace all backslash in the content string with nothing
            content = content.replace("\\", "")

            info_dict_raw = ast.literal_eval(content)

            info_dict = {}
            for (key, value) in info_dict_raw.items():
                if is_empty_string_nan_or_none(value):
                    info_dict[key] = ""
                else:
                    info_dict[key] = value

            print(f"Successfully parsed LLM output. ({content=})")
        except Exception:
            print(
                f"Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. ({content=})")
            print(f"Will ignore the error for report {i} and continue.")
            if debug:
                breakpoint()
            info_dict = {}
            error_count += 1

            # raise Exception(f"Failed to parse LLM output. Did you set --n_predict too low or is the input too long? Maybe you can try to lower the temperature a little. ({content=})") from e

        # get metadata from df by looking for row where id == id and get the column metadata

        metadata = df[df['id'] == id]['metadata'].iloc[0]
        import ast
        metadata = ast.literal_eval(metadata)
        metadata['llm_processing'] = llm_metadata

        import json

        # Construct a dictionary containing the report and extracted information
        extracted_info = {'report': info['report'],
                          'id': id, 'metadata': json.dumps(metadata)}
        for key, value in info_dict.items():
            extracted_info[key] = value

        # Append the extracted information to the list
        extracted_data.append(extracted_info)

    # Convert the list of dictionaries into a DataFrame
    df = pd.DataFrame(extracted_data)

    def extract_base_id(id):
        parts = id.split('$')
        base_id = parts[0]  # The part before the dollar sign

        if len(parts) > 1:  # If there's a dollar sign in the ID
            subparts = parts[1].split('_')
            if len(subparts) > 1 and subparts[-1].isdigit():
                # If there's an underscore followed by a number after the dollar sign
                return base_id + '$' + '_'.join(subparts[:-1])

        return id  # Return the original ID if no underscore followed by a number is found after the dollar sign

    df['base_id'] = df['id'].apply(extract_base_id)
    # test_id = 'ocr_arztbericht-bild.pdf$eea5469f-f6a4-4b08-92f3-2340c61b0745'
    # breakpoint()

    # df['base_id'] = df['id'].apply(lambda x: '_'.join(x.split('_')[:-1]) if '_' in x else x)

    # Group by base_id and aggregate reports and other columns into lists
    aggregated_df = df.groupby('base_id').agg(
        lambda x: x.tolist() if x.name != 'report' else ' '.join(x)).reset_index()

    aggregated_df['personal_info_list'] = aggregated_df.apply(lambda row: [item for list in row.drop(
        ["id", "base_id", "report", "metadata"]) for item in list], axis=1)

    aggregated_df['masked_report'] = aggregated_df['report'].apply(
        lambda x: replace_personal_info(x, aggregated_df['personal_info_list'][0], []))

    aggregated_df.drop(columns=['id'], inplace=True)
    aggregated_df.rename(columns={'base_id': 'id'}, inplace=True)

    aggregated_df['metadata'] = aggregated_df['metadata'].apply(lambda x: x[0])

    return aggregated_df, error_count


def get_context_size(yaml_filename, model_name):
    import yaml
    with open(yaml_filename, 'r') as file:
        config_data = yaml.safe_load(file)

    print(config_data)
    print("Find ", model_name)

    for model in config_data["models"]:
        if model["path_to_gguf"] == model_name:
            return int(model["context_size"])

    print("FAILED TO FIND CONTEXT SIZE")
    return None  # Model not found in the YAML file


@llm_processing.route("/llm", methods=['GET', 'POST'])
def main():

    form = LLMPipelineForm(
        current_app.config['CONFIG_FILE'], current_app.config['MODEL_PATH'])
    form.variables.render_kw = {'disabled': 'disabled'}

    if form.validate_on_submit():
        file = request.files["file"]

        if file.filename.endswith('.csv'):
            try:
                print(file)
                df = pd.read_csv(file)
            except pd.errors.ParserError as e:
                # print the error message in console
                print(e)
                print("The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed. Usually, this is caused by a line break in a field. The file will be fixed and then read again.")
                # fix the file
                fixed_file = BytesIO()
                read_and_save_csv(file, fixed_file)
                fixed_file.seek(0)
                df = pd.read_csv(fixed_file)

        elif file.filename.endswith('.xlsx'):
            try:
                df = pd.read_excel(file)
                print(df.head())
                # ValueError: Excel file format cannot be determined, you must specify an engine manually.
            except ValueError as e:
                print(e)
                print("The error message indicates that the Excel file format cannot be determined. This means that the Excel file is not properly formatted and needs to be fixed. The file will be fixed and then read again.")
                # fix the file
                flash("Excel file is not properly formatted!", "danger")
                return render_template("llm_processing.html", form=form)

        elif file.filename.endswith('.zip'):

            zip_buffer = BytesIO()
            file.save(zip_buffer)
            zip_buffer.seek(0)

            temp_dir = tempfile.mkdtemp()

            # Save the uploaded file to the temporary directory
            zip_file_path = os.path.join(temp_dir, file.filename)
            with open(zip_file_path, 'wb') as f:
                f.write(zip_buffer.getvalue())
                print("Zip file saved:", zip_file_path)

            # Verify the integrity of the saved file (optional)
            if os.path.exists(zip_file_path):
                saved_file_size = os.path.getsize(zip_file_path)
                print(f"Saved file size: {saved_file_size} bytes")

                # Check if the saved file is a valid ZIP file
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as test_zip:
                        test_zip.testzip()
                    print("File is a valid ZIP file")
                except zipfile.BadZipFile:
                    print("File is not a valid ZIP file")

            else:
                print("File not found:", zip_file_path)

            # Now you can proceed to read the contents of the ZIP file
            df = read_preprocessed_csv_from_zip(zip_file_path)

            if df is None:
                flash(
                    "Zip file seems to be malformed or in a not supported format! Is there a csv file in it?", "danger")
                return render_template("llm_processing.html", form=form)

        else:
            flash("File format not supported!", "danger")
            return render_template("llm_processing.html", form=form)

        model_name = ""

        for filename, name in form.model.choices:
            if filename == form.model.data:
                model_name = name

        variables = [var.strip() for var in form.variables.data.split(",")]

        current_datetime = datetime.now()
        prefix = current_datetime.strftime("%Y%m%d%H%M")

        job_id = model_name.replace(" ", "").replace("_", "-") + "_" + prefix + "_" + secrets.token_urlsafe(8)

        if not os.path.exists(current_app.config['SERVER_PATH']):
            flash(
                "Llama CPP Server executable not found. Did you specify --server_path correctly?", "danger")
            return render_template("llm_processing.html", form=form)

        print("Run job!")
        global llm_jobs

        # extract_from_report(
        #     df=df,
        #     model_name=form.model.data,
        #     prompt=form.prompt.data,
        #     symptoms=variables,
        #     temperature=float(form.temperature.data),
        #     grammar=form.grammar.data.replace("\r\n", "\n"),
        #     model_path=current_app.config['MODEL_PATH'],
        #     server_path=current_app.config['SERVER_PATH'],
        #     n_predict=current_app.config['N_PREDICT'],
        #     ctx_size=current_app.config['CTX_SIZE'],
        #     n_gpu_layers=current_app.config['N_GPU_LAYERS'],
        #     job_id=job_id,
        #     zip_file_path=zip_file_path or None
        # )

        update_progress(job_id=job_id, progress=(0, len(df), True))

        llm_jobs[job_id] = executor.submit(
            extract_from_report,
            df=df,
            model_name=form.model.data,
            prompt=form.prompt.data,
            symptoms=variables,
            temperature=float(form.temperature.data),
            grammar=form.grammar.data.replace("\r\n", "\n"),
            model_path=current_app.config['MODEL_PATH'],
            server_path=current_app.config['SERVER_PATH'],
            n_predict=current_app.config['N_PREDICT'],
            ctx_size=get_context_size(
                current_app.config['CONFIG_FILE'], form.model.data),
            n_gpu_layers=current_app.config['N_GPU_LAYERS'],
            job_id=job_id,
            zip_file_path=zip_file_path or None,
            llamacpp_port=current_app.config['LLAMACPP_PORT'],
            debug=current_app.config['DEBUG'],
            model_name_name = model_name
        )

        print("Started job successfully!")

        return redirect(url_for('llm_processing.llm_results'))

    return render_template("llm_processing.html", form=form)


@llm_processing.route("/llm_results", methods=['GET'])
def llm_results():

    global llm_progress
    return render_template("llm_results.html", llm_progress=llm_progress, model_loaded=not new_model)


@llm_processing.route("/llm_download", methods=['GET'])
def llm_download():
    job_id = request.args.get('job')

    if job_id not in llm_jobs:
        flash("Job not found!", "danger")
        return redirect(url_for('llm_processing.llm_results'))

    job = llm_jobs[job_id]

    if job.done():
        try:
            (result_df, error_count), zip_file_path = job.result()
        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for('llm_processing.llm_results'))

        if not zip_file_path or not os.path.exists(zip_file_path):
            print("Download only the csv.")
            result_io = BytesIO()
            # breakpoint()

            result_df.to_csv(result_io, index=False)
            result_io.seek(0)
            return send_file(
                result_io,
                mimetype="text/csv",
                as_attachment=True,
                download_name=f"llm-output-{job_id}.csv",
            )

        with zipfile.ZipFile(zip_file_path, "r") as existing_zip:
            # Create an in-memory BytesIO object to hold the updated ZIP file
            updated_zip_buffer = io.BytesIO()

            # Create a new ZIP file
            with zipfile.ZipFile(updated_zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as updated_zip:
                # Add all existing files from the original ZIP
                for existing_file in existing_zip.filelist:
                    updated_zip.writestr(
                        existing_file.filename, existing_zip.read(existing_file.filename))

                # Add the DataFrame as a CSV file to the ZIP
                csv_buffer = io.StringIO()
                result_df.to_csv(csv_buffer, index=False)
                updated_zip.writestr(
                    f"llm-output-{job_id}.csv", csv_buffer.getvalue())

        # Reset the BytesIO object to the beginning
        updated_zip_buffer.seek(0)

        # Send the updated ZIP file
        return send_file(
            updated_zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"llm-output-{job_id}.zip"
        )

    else:
        flash(f"Job {job}: An unknown error occurred! Probably the model did not predict anything / the output is empty and / or the code ran into a breakpoint!", "danger")
        return redirect(url_for('llm_processing.llm_results'))
