Educ.

1. Project Overview

Description:
A Flask-based platform designed to make learning more practical and inclusive, addressing the challenges of theoretical learning and accessibility for students with disabilities.

Inspiration:
We chose this idea to bridge the gap between theoretical education and practical application. Our goal is to foster an inclusive environment where every student, regardless of ability, can learn interactively and effectively.
SDG 4: Quality Education 
Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all.
SDG 10: Reduced Inequalities
By supporting disabled individuals and offering inclusive education, your website helps bridge learning gaps and promote equal opportunities.


2. Features

Interactive Chemistry Courses: 
    -Virtual simulations for conducting chemical reactions safely and effectively.
AR-Based Chemistry Revision: 
    -View and manipulate 3D models of atoms and molecules using augmented reality (AR).
Accessibility Services Module:
    -Real-time sign language detection converted to speech and text.
    -Voice-to-text and text-to-voice conversion tools.
    -A virtual keyboard for enhanced accessibility.

3. Technology Stack

Frontend: HTML, CSS, JavaScript, Unity (for AR).
Backend: Flask (Python).
Other Tools: OpenCV (for sign language detection), Unity AR toolkit.

4. Installation and Usage

Prerequisites

Python 3.9+
Unity 2021.3.28f1 (LTS)/Google ARCore & Apple ARKit
Core Libraries:
    -Django (4.2)
    -Flask (3.1.0)
    -TensorFlow (2.18.0)
    -Keras (3.7.0)
    -NumPy (1.26.4)
    -Pandas (2.2.3)
    -Scikit-learn (1.6.1)
    -OpenAI (1.59)
    -Seaborn (0.13.2)
Utility Libraries:
    -Requests (2.32.3)
    -BeautifulSoup4 (4.12.3)
    -Pillow (10.0.0)
    -PyYAML (6.0.2)
    -SQLAlchemy
    -Werkzeug (3.1.3)
    -Flask-SocketIO (5.5.1)
    -python-dotenv (1.0.1)
Data Science & Machine Learning:
    -Jax (0.5.0)
    -SciPy (1.15.1)
    -Matplotlib (3.10.0)
Other Libraries:
    -pyttsx3 (2.98)
    -SpeechRecognition (3.14.0)
    -opencv-python (4.11.0.86)

Installation Steps

Clone the repository:
    git clone https://github.com/Roua-Khalfet/Education_project.git
Navigate to the project directory:
    cd Education_project
Install dependencies:
    pip install -r requirements.txt
Run the application:
    python app.py



5. Usage Instructions

Step 1: Open the application in your browser at http://127.0.0.1:5000.
Step 2: Navigate to the desired module.
Step 3: Follow the on-screen instructions to explore each feature.




6. Demo
Screenshots:
1) Homepage
![image](https://github.com/user-attachments/assets/7067165c-47c7-48aa-9508-581d05f0a00e)
2) Nos cours
![image](https://github.com/user-attachments/assets/f235ac58-43f0-4820-938c-46d38003f0f3)
3) French test using text to speech and speech to text
![image](https://github.com/user-attachments/assets/58edab14-80c7-4448-91aa-85a27b5affe6)
4) Creating chemical reactions by simulating atomic interactions
![image](https://github.com/user-attachments/assets/95dc1f3c-49e7-4b26-aa6e-e045d91c826d)
![Uploading image (1).png…]()
5) Visualizing molecules via 3D atomes
![image](https://github.com/user-attachments/assets/f9f9fa8c-bbc6-467a-b079-63f425f1b2da)
6) Accessible services for disabled people
![image](https://github.com/user-attachments/assets/6c86431c-4d2f-45f4-8613-bc93f2cdfa27)
7) Translating sign language to text 
![image](https://github.com/user-attachments/assets/33b0fbc6-8184-4c67-ac67-c90546cb3902)
8) Virual keyboards 
![image](https://github.com/user-attachments/assets/a37a4f2c-a61d-4891-b29f-f2d1bd02b996)


8. Team Members

Nour Jazi: Frontend Developer.
Douaa Boudokhan , Youssef Ben Lallahom , Roua Khalfet: Backend Developer.
Douaa Boudokhan , Youssef Ben Lallahom , Roua Khalfet: Accessibility Tools Specialist.


8. Challenges Faced
AR Integration: Integrating smooth AR experiences using Unity and ensuring compatibility with various devices.
Sign Language Detection: Training accurate models for real-time detection using OpenCV.
Virtual Keyboard with Eye Tracking: Navigate with eye movements and select keys with a wink for enhanced accessibility.
Accessibility: Ensuring features are user-friendly and functional for students with diverse needs.

Solutions:
We adopted iterative testing and user feedback for AR.
Optimized the sign language detection model with dataset improvements.
Focused on UI/UX design for accessibility tools.

9. Future Improvements

Expanding the AR Library: Adding more complex molecules and reactions to enhance the chemistry learning experience.
Introducing More Subjects: Expanding the platform to include practical and interactive learning tools for other subjects like physics, biology, and mathematics.
Multilingual Support: Adding support for more languages in the accessibility module to reach a broader audience.
Gamification: Incorporating gamified elements such as quizzes, badges, and leaderboards to boost engagement and motivation.

10. License

This project is developed solely for demonstration purposes as part of a hackathon.
All rights reserved by the creators: Vortex Team.

If you'd like to use or adapt this project, please contact us for permission.

11. Acknowledgments

Mentors and sponsors of the hackathon.
Resources like OpenCV, Flask documentation, and Unity toolkit.
Open-source contributors who inspired this project.

12. Datasets
https://www.kaggle.com/datasets/hojjatk/mnist-dataset
