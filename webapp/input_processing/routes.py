from datetime import datetime
import shutil
from flask import render_template, request, redirect, url_for, flash, send_file, session
import os
import tempfile
from werkzeug.utils import secure_filename
from .forms import PreprocessUploadForm
import secrets
from concurrent import futures
import pandas as pd
import pdfplumber
import time
import subprocess
from PIL import Image
from docx import Document
from docx2pdf import convert
from odf import teletype
from odf.opendocument import load
import uuid
import zipfile
from io import BytesIO
from fpdf import FPDF
from . import input_processing
from .. import socketio


JobID = str
jobs: dict[JobID, futures.Future] = {}
executor = futures.ThreadPoolExecutor(1)

job_progress = {}


@socketio.on('connect')
def handle_connect():
    print("Client Connected")


@socketio.on('disconnect')
def handle_disconnect():
    print("Client Disconnected")


def update_progress(job_id, progress: tuple[int, int, bool]):
    global job_progress
    job_progress[job_id] = progress

    print("Progress: ", progress)
    socketio.emit('progress_update', {
                  'job_id': job_id, 'progress': progress[0], 'total': progress[1]})


def failed_job(job_id):
    time.sleep(2)
    print("FAILED")
    global job_progress
    # wait for 1s
    socketio.emit('progress_failed', {'job_id': job_id})


def complete_job(job_id):
    print("COMPLETE")
    global job_progress
    socketio.emit('progress_complete', {'job_id': job_id})


def save_text_as_pdf(text, pdf_file_save_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 6, txt=text)

    pdf.output(pdf_file_save_path)

def convert_docx_to_pdf(docx_file_path, pdf_file_path=None):
    convert(docx_file_path, pdf_file_path)


def preprocess_input(job_id, file_paths):
    print("PREPROCESS")

    merged_data = []
    for i, file_path in enumerate(file_paths):
        try:
            # if file_path.endswith('.csv'):
            #     df = pd.read_csv(file_path)
            #     merged_data.append(df)
            #     print("CSV to pdf conversion is not supported yet.")
            if file_path.endswith(('.pdf', '.jpg', '.jpeg', '.png')):
                if not file_path.endswith('.pdf'):
                    # Convert JPG/PNG to PDF
                    pdf_output_path = os.path.join(tempfile.mkdtemp(), f"pdf_{os.path.basename(file_path)}.pdf")
                    image = Image.open(file_path)
                    image.save(pdf_output_path)
                    file_path = pdf_output_path

                # Run OCRmyPDF
                contains_text = False
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        if len(page.extract_text()) > 0:
                            contains_text = True
                            break

                if not contains_text:
                    ocr_output_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path)}")
                    if shutil.which("tesseract") is not None:
                        if shutil.which("ocrmypdf") is not None:
                            # Command exists, proceed with subprocess
                            subprocess.run(
                                ['ocrmypdf', '-l', 'deu', '--force-ocr', file_path, ocr_output_path])
                        else:
                            print(f"OCRMyPDF not found, skipping OCR for {file_path}")
                            return "OCRMyPDF not found but required for OCR."
                    else:
                        print(f"Tesseract not found, skipping OCR for {file_path}")
                        return "Tesseract not found but required for OCR."

                else:
                    ocr_output_path = file_path

                with pdfplumber.open(ocr_output_path) as ocr_pdf:
                    ocr_text = ''
                    for page in ocr_pdf.pages:
                        ocr_text += page.extract_text()
                print("Save Report as ", ocr_output_path)
                merged_data.append(pd.DataFrame(
                    {'report': [ocr_text], 'filepath': ocr_output_path}))

            elif file_path.endswith('.txt'):
                with open(file_path, 'r') as f:
                    text = f.read()
                    pdf_file_save_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path).split('.txt')[0]}.pdf")
                    
                    save_text_as_pdf(text, pdf_file_save_path)

                    merged_data.append(pd.DataFrame({'report': [text], 'filepath': pdf_file_save_path}))
            elif file_path.endswith('.docx'):
                doc = Document(file_path)
                doc_text = '\n'.join(
                    [paragraph.text for paragraph in doc.paragraphs])
                
                pdf_file_save_path = os.path.join(tempfile.mkdtemp(), f"ocr_{os.path.basename(file_path).split('.docx')[0]}.pdf")
                
                convert_docx_to_pdf(file_path, pdf_file_save_path)

                merged_data.append(pd.DataFrame({'report': [doc_text], 'filepath': pdf_file_save_path}))
            # elif file_path.endswith('.odt'):
            #     doc = load(file_path)
            #     doc_text = ''
            #     for element in doc.getElementsByType(text.P):
            #         doc_text += teletype.extractText(element)
            #     merged_data.append(pd.DataFrame({'report': [doc_text]}))
            else:
                print(f"Unsupported file format: {file_path}")
        except Exception as e:
            print(f"Error processing file {file_path} (might also be that tesseract / ghostscript / ocrmypdf is not installed or not in PATH): {e}")
            update_progress(job_id=job_id, progress=(
                i, len(file_paths), False))
            os.remove(file_path)
            return "Error processing file: " + str(e)

        update_progress(job_id=job_id, progress=(i+1, len(file_paths), True))

    merged_df = pd.concat(merged_data)
    complete_job(job_id)
    return merged_df


@input_processing.route("/download", methods=['GET'])
def download():
    job_id = request.args.get("job")
    global jobs

    job = jobs[job_id]

    if job.cancelled():
        flash(f"Job {job} was cancelled", "danger")
        return redirect(url_for('input_processing.main'))
    elif job.running():
        flash(f"Job {job} is still running", "warning")
        return redirect(url_for('input_processing.main'))
    elif job.done():
        try:
            df = job.result()
        except Exception:
            flash("Preprocessing failed / did not output anything useful!", "danger")
            return redirect(url_for('input_processing.main'))

        if isinstance(df, str):
            flash(df, "danger")
            return redirect(url_for('input_processing.main'))

        # split the text in chunks
        max_length = session['text_split']

        # Add an 'id' column and generate unique IDs for every row
        # df['id'] = df.apply(lambda x: str(uuid.uuid4()), axis=1)

        def remove_ocr_prefix(filename):
            if filename.startswith('ocr_'):
                return filename[len('ocr_'):]
            else:
                return filename

        df['filename'] = df['filepath'].apply(lambda x: remove_ocr_prefix(os.path.basename(x)))
        df['id'] = df.apply(lambda x: x['filename'] +
                            '$' + str(uuid.uuid4()), axis=1)

        # add metadata column with json structure. Add the current date and time as preprocessing key in the json structure
        df['metadata'] = df.apply(lambda x: {'preprocessing': {
                                  'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}}, axis=1)

        # Optionally, you can drop the 'filename' column if you don't need it anymore
        df.drop(columns=['filename'], inplace=True)

        # Function to add files to a zip file

        def add_files_to_zip(zipf, files, ids):
            for file, file_id in zip(files, ids):
                zipf.write(file, f"{file_id}.{os.path.basename(file).split('.')[-1]}")
                # os.remove(file)sss

        # Add dataframe as CSV to zip
        def add_dataframe_to_zip(zipf, df):
            # Create a temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                csv_filename = f'preprocessed_{job_id}.csv'
                csv_filepath = os.path.join(temp_dir, csv_filename)

                # Drop unnecessary columns and save the dataframe to a CSV file
                df.drop(columns=['filepath'], inplace=True)
                df.to_csv(csv_filepath, index=False)

                # Write the CSV file to the zip archive
                zipf.write(csv_filepath, arcname=csv_filename)

        files_to_zip = df['filepath'].tolist()
        ids = df['id'].tolist()

        # Split rows containing more than max_length letters
        split_rows = []
        for index, row in df.iterrows():
            if len(row['report']) > max_length:
                num_splits = (len(row['report']) +
                              max_length - 1) // max_length
                for i in range(num_splits):
                    split_row = row.copy()
                    split_row['report'] = row['report'][i *
                                                        max_length: (i + 1) * max_length]
                    split_row['id'] = f'{row["id"]}_{i}'
                    split_rows.append(split_row)
            else:
                split_rows.append(row)

        # Create a new DataFrame with the split rows
        df_split = pd.DataFrame(split_rows)

        print("Files to zip:", files_to_zip)

        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            add_files_to_zip(zipf, files_to_zip, ids)
            add_dataframe_to_zip(zipf, df_split)

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"preprocessed-{job_id}.zip",
        )
    else:
        flash(f"Job {job}: An unknown error occurred!", "danger")
        return redirect(url_for('input_processing.main'))


@input_processing.route("/", methods=['GET', 'POST'])
def main():
    print("MAIN")

    form = PreprocessUploadForm()

    if form.validate_on_submit():

        current_datetime = datetime.now()
        prefix = current_datetime.strftime("%Y%m%d%H%M")

        job_id = f"{form.text_split.data}-{prefix}-" + secrets.token_urlsafe(8)

        temp_dir = tempfile.mkdtemp()

        session['text_split'] = form.text_split.data

        # Save each uploaded file to the temporary directory
        file_paths = []
        for file in form.files.data:
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                print("File saved:", file_path)
                file_paths.append(file_path)

        update_progress(job_id=job_id, progress=(
            0, len(form.files.data), True))

        global jobs
        jobs[job_id] = executor.submit(
            preprocess_input,
            job_id=job_id,
            file_paths=file_paths
        )

        flash('Upload Successful!', "success")
        return redirect(url_for('input_processing.main'))

    global job_progress

    return render_template("index.html", title="LLM Anonymizer", form=form, progress=job_progress)
