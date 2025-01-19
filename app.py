from flask import Flask, render_template,request
import subprocess
import sys
import os



from camerahands import SignLanguageDetector  # Importez votre classe ici

app = Flask(__name__)

# Initialisation de l'objet SignLanguageDetector
detector = SignLanguageDetector()


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route('/courses')
def courses():
    return render_template('courses.html')

@app.route('/revision')
def revision():
    return render_template('revision.html')

@app.route('/teacher')
def teacher():
    return render_template('teacher.html')

@app.route('/chimie')
def chimie():
    return render_template('chimie.html')

@app.route('/french')
def french():
    return render_template('french.html')

@app.route('/notes')
def notes():
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    keyboard_script = os.path.join(current_dir, 'virtual_keyboard.py')

    try:
        # Run the virtual keyboard script and capture its output
        result = subprocess.run(
            [sys.executable, keyboard_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout if result.stdout else result.stderr
        return render_template('notes.html', output=output)
    except Exception as e:
        return render_template('notes.html', output=f"Error launching keyboard: {str(e)}")
    

@app.route('/sign')
def sign():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    camera_script = os.path.join(current_dir, 'camerahands.py')
    
    try:
        # Run camerahands.py and capture its output
        process = subprocess.Popen([sys.executable, camera_script], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE,
                                 text=True)
        
        # Get the output
        output, _ = process.communicate()
        
        # Find the final results in the output
        final_results = ""
        for line in output.split('\n'):
            if "Words formed:" in line:
                final_results = line.strip()
        
        return render_template('sign.html', final_results=final_results)
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)