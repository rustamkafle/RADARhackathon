from flask import Flask, render_template, request
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__)

# Path to your folder containing Excel files
FOLDER_PATH = "."  # Change this to your folder path if needed

# Disable caching to always read fresh data
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def clean_column_name(col):
    """Clean column names to show only date without time"""
    if isinstance(col, datetime):
        return col.strftime('%Y-%m-%d')
    elif isinstance(col, str):
        # Try to parse and format if it looks like a date
        try:
            dt = pd.to_datetime(col)
            return dt.strftime('%Y-%m-%d')
        except:
            return col
    return col

def read_attendance(subject):
    """Read attendance data from Excel file"""
    try:
        file_path = os.path.join(FOLDER_PATH, f"{subject}.xlsx")
        df = pd.read_excel(file_path)
        
        # Clean column names to remove time portion
        df.columns = [clean_column_name(col) for col in df.columns]
        
        # Convert DataFrame to HTML table
        html_table = df.to_html(classes='attendance-table', index=False, na_rep='-')
        return html_table
    except Exception as e:
        return f"<p class='error'>Error reading file: {str(e)}</p>"

@app.route('/')
def home():
    """Home page with subject buttons"""
    return render_template('index.html')

@app.route('/view/<subject>')
def view_attendance(subject):
    """View attendance for selected subject"""
    attendance_html = read_attendance(subject)
    return render_template('view.html', subject=subject, attendance=attendance_html)

@app.after_request
def add_header(response):
    """Disable caching to ensure fresh data"""
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

# Create templates directory if it doesn't exist
if not os.path.exists('templates'):
    os.makedirs('templates')

# Write index.html
with open('templates/index.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Attendance Viewer</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <style>
        body {
            font-family: Arial, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            background: white;
            padding: 40px;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
            text-align: center;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .button-group {
            display: flex;
            gap: 20px;
            justify-content: center;
        }
        .subject-btn {
            padding: 15px 40px;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: all 0.3s;
            color: white;
            text-decoration: none;
            display: inline-block;
        }
        .subject-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .subject1-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .subject2-btn {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📚 Attendance Viewer</h1>
        <div class="button-group">
            <a href="/view/subject1" class="subject-btn subject1-btn">Subject 1</a>
            <a href="/view/subject2" class="subject-btn subject2-btn">Subject 2</a>
        </div>
    </div>
</body>
</html>
''')

# Write view.html
with open('templates/view.html', 'w') as f:
    f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>{{ subject|title }} Attendance</title>
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.2);
        }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        h1 {
            color: #333;
            margin: 0;
        }
        .back-btn {
            padding: 10px 25px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s;
        }
        .back-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .refresh-btn {
            padding: 10px 25px;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            text-decoration: none;
            display: inline-block;
            transition: all 0.3s;
            margin-left: 10px;
        }
        .refresh-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .attendance-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .attendance-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        .attendance-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #ddd;
        }
        .attendance-table tr:hover {
            background-color: #f5f5f5;
        }
        .attendance-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .error {
            color: #f5576c;
            padding: 20px;
            background: #ffe5e9;
            border-radius: 5px;
        }
        .button-group {
            display: flex;
            gap: 10px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ subject|title|replace('subject', 'Subject ')|replace('1', '1')|replace('2', '2') }} Attendance</h1>
            <div class="button-group">
                <a href="/" class="back-btn">← Back to Subjects</a>
                <a href="/view/{{ subject }}" class="refresh-btn" onclick="location.reload();">🔄 Refresh</a>
            </div>
        </div>
        {{ attendance|safe }}
    </div>
</body>
</html>
''')

if __name__ == '__main__':
    print("Starting Flask Attendance Viewer...")
    print("Open your browser and go to: http://127.0.0.1:5000")
    print("\n⚡ Auto-refresh enabled - changes to Excel files will be reflected immediately!")
    app.run(debug=True, port=5000)
