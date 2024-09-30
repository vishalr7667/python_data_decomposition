# import os
# import pandas as pd
# from flask import Flask, render_template, request, flash, session, redirect, url_for
# from werkzeug.utils import secure_filename

# app = Flask(__name__)
# app.secret_key = "supersecretkey"
# app.config['UPLOAD_FOLDER'] = 'uploads'
# app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
# app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # This sets the max file size to 100 MB

# # Ensure the upload folder exists
# if not os.path.exists(app.config['UPLOAD_FOLDER']):
#     os.makedirs(app.config['UPLOAD_FOLDER'])

# # Allowed file extensions check
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Clear the session to avoid old data persisting
#         session.pop('filtered_columns', None)
#         session.pop('correlation_matrix', None)
#         session.pop('selected_columns', None)

#         # Handle file upload
#         if 'file' in request.files:
#             file = request.files['file']
#             if file.filename == '':
#                 flash('No file selected')
#                 return redirect(request.url)

#             if file and allowed_file(file.filename):
#                 filename = secure_filename(file.filename)
#                 filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#                 file.save(filepath)

#                 try:
#                     # Read file based on extension
#                     if filename.endswith('.csv'):
#                         df = pd.read_csv(filepath)
#                     elif filename.endswith('.xlsx'):
#                         df = pd.read_excel(filepath)

#                     # Preprocess the data
#                     df_cleaned = preprocess_data(df)
                    
#                     # Perform correlation analysis
#                     correlation_matrix = perform_correlation(df_cleaned)
#                     session['correlation_matrix'] = correlation_matrix.to_dict()

#                     # Filter columns by correlation
#                     filtered_columns = filter_columns_by_correlation(correlation_matrix)
#                     session['filtered_columns'] = filtered_columns
                    
#                     flash('File uploaded and analyzed. Please select variables for further analysis.')
#                     return redirect(url_for('index'))

#                 except Exception as e:
#                     flash(f"Error reading file: {e}")
#                     return redirect(request.url)

#     columns = session.get('filtered_columns', [])
    
#     return render_template('index.html', columns=columns)

# @app.route('/select_variables', methods=['POST'])
# def select_variables():
#     # Handle variable selection
#     selected_columns = request.form.getlist('columns')
#     session['selected_columns'] = selected_columns
    
#     return redirect(url_for('decomposition'))

# @app.route('/decomposition')
# def decomposition():
#     # Get the selected variables for decomposition
#     selected_columns = session.get('selected_columns', [])
    
#     if not selected_columns:
#         flash('No variables selected for decomposition.')
#         return redirect(url_for('index'))
    
#     return render_template('decomposition.html', selected_columns=selected_columns)

# # Data preprocessing (handling missing values, outliers, etc.)
# def preprocess_data(df):
#     df_cleaned = df.dropna()  # Simple method, you can expand this
#     return df_cleaned

# # Perform correlation analysis (Pearson correlation)
# def perform_correlation(df):
#     numerical_columns = df.select_dtypes(include=['number']).columns
#     correlation_matrix = df[numerical_columns].corr(method='pearson')
#     return correlation_matrix

# # Filter columns based on a correlation threshold
# def filter_columns_by_correlation(correlation_matrix, threshold=0.5):
#     high_correlation_columns = correlation_matrix.columns[correlation_matrix.abs().max() > threshold].tolist()
#     return high_correlation_columns

# @app.route('/clear_session', methods=['POST'])
# def clear_session():
#     # Clear session data
#     session.pop('filtered_columns', None)
#     session.pop('correlation_matrix', None)
#     session.pop('selected_columns', None)
#     return '', 204  # Return a 204 No Content response

# if __name__ == '__main__':
#     app.run(debug=True)


import os
import pandas as pd
from flask import Flask, render_template, request, flash, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'csv', 'xlsx'}
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB file size limit

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Handle file size limit error
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(error):
    flash('File size exceeds the allowed limit of 100 MB.')
    return redirect(url_for('index'))

# Allowed file extensions check
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload
        if 'file' not in request.files:
            flash('No file part in the request.')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No file selected')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)

                # Read file based on extension
                if filename.endswith('.csv'):
                    try:
                        df = pd.read_csv(filepath, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(filepath, encoding='ISO-8859-1')  # Fallback encoding
                elif filename.endswith('.xlsx'):
                    df = pd.read_excel(filepath)

                # Preprocess the data
                df_cleaned = preprocess_data(df)
                
                # Perform correlation analysis
                correlation_matrix = perform_correlation(df_cleaned)
                if correlation_matrix.empty:
                    flash('No numerical columns for correlation analysis.')
                    return redirect(request.url)
                
                # Filter columns by positive correlation
                filtered_columns = filter_columns_by_correlation(correlation_matrix)
                
                # Debug: Print the filtered columns to verify filtering
                print(f"Filtered columns (positive correlation): {filtered_columns}")  # Debug line

                # Check if there are filtered columns
                if not filtered_columns:
                    flash('No columns found with significant positive correlation.')
                    return redirect(request.url)

                session['filtered_columns'] = filtered_columns

                flash('File uploaded and analyzed. Highly correlated columns selected.')
                return redirect(url_for('index'))

            except Exception as e:
                flash(f"Error reading file: {e}")
                app.logger.error(f"Error occurred while processing the file: {e}", exc_info=True)
                return redirect(request.url)

    # Only show filtered (positively correlated) columns
    columns = session.get('filtered_columns', [])
    return render_template('index.html', columns=columns)

@app.route('/select_variables', methods=['POST'])
def select_variables():
    selected_columns = request.form.getlist('columns')
    session['selected_columns'] = selected_columns
    return redirect(url_for('decomposition'))

@app.route('/decomposition')
def decomposition():
    selected_columns = session.get('selected_columns', [])
    if not selected_columns:
        flash('No variables selected for decomposition.')
        return redirect(url_for('index'))
    return render_template('decomposition.html', selected_columns=selected_columns)

def preprocess_data(df):
    # Optionally: Add more preprocessing steps like scaling, outlier removal, etc.
    df_cleaned = df.dropna()  # Simple method, you can expand this
    return df_cleaned

def perform_correlation(df):
    numerical_columns = df.select_dtypes(include=['number']).columns
    if numerical_columns.empty:
        flash("No numerical columns available for correlation analysis.")
        return pd.DataFrame()
    
    correlation_matrix = df[numerical_columns].corr(method='pearson')
    
    # Debug: Print the correlation matrix to verify correctness
    print(f"Correlation matrix:\n{correlation_matrix}")  # Debug line
    
    return correlation_matrix

def filter_columns_by_correlation(correlation_matrix, threshold=0.5):
    """
    Filter only columns that have positive correlation greater than the specified threshold.
    """
    positive_correlation_columns = []
    
    # Iterate over each column and check the correlation values
    for column in correlation_matrix.columns:
        # Find the maximum positive correlation for each column (ignoring self-correlation which is always 1)
        max_correlation = correlation_matrix[column][correlation_matrix[column] < 1].max()  # Exclude self-correlation
        
        # Debug: Print the max correlation value for each column
        print(f"Column '{column}' max positive correlation: {max_correlation}")  # Debug line
        
        # Include only columns with positive correlation higher than the threshold
        if max_correlation > threshold:
            positive_correlation_columns.append(column)
    
    # Debug: Print the final list of columns with positive correlation
    print(f"Final list of positively correlated columns: {positive_correlation_columns}")  # Debug line
    
    return positive_correlation_columns

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
