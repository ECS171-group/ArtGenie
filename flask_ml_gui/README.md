#### Basic Flask Tutorial for Project GUI
This is a basic tutorial to show an example of how you can begin development for the GUI using Flask. Using Flask or following these instructions is not required. You should find online documentation of Flask, html and css to be helpful. When choosing a web framework if you don't have prior experience, it is best to choose one better suited for beginners such as Flask or Express where you have access to documentation and guides.

Keep in mind that the project description include developing a basic web-based front-end to invoke and run the model(s) on input data and display the prediction output. While this front-end can be quite simple, we are looking for it to be able to input data easily and display the appropriate output from your model(s). Also, while the project report rubric does not explicitly mention the project GUI, it is something you can briefly mention in your methodology and should be covered in your project roadmap. Be sure to have the working GUI in your github submission. If you have any more questions feel free to ask during office hours.


##### Prerequisites:
- Python3
- Code Editor
- Basic understanding of HTML, CSS (for styling), python

##### Steps:
1. Create a Project Directory
   - Create a new directory for your project: `mkdir flask_ml_gui`
   - Change directory into the project folder: `cd flask_ml_gui`
   - Create a new directory `mkdir templates` to store html templates
2. Set up a Virtual Environment (Optional)
   - Virtual environments isolate project dependencies, avoiding conflicts with other projects on your system. 
   - Use `python -m venv venv` to create a virtual environment named `venv`.
   - Activate the virtual environment
     - For Windows: `venv\Scripts\activate.bat`
     - For macOS/Linux: `source venv/bin/activate`
3. Install Flask
   - Use `pip install Flask` within your activated virtual environment or terminal to install the Flask framework.  
4. Create Flask application
   - In your project directory, create a new Python file named `app.py`
   - Your Flask application will need to define define Flask routes to handle incoming requests from the HTML form.
   - Create a route to serve the HTML template (index.html) when the user visits the homepage.
   - Create another route to handle form submission and process the user input.
   - You can import a saved ML model or train an ML and define a function to perform the machine learning task using the input provided by the user.
   - Here is a basic template for `app.py` that includes an `index` and `predict` function with corresponding routes - note that commenting out `import your_ml_model`, `model = your_ml_model.load_model()` and the entire `def predict()` function should allow this template to run out of the box, and when you finish the corresponding sections, you can include them in the web app.
    ```Python
    from flask import Flask, render_template, request
    import your_ml_model  # Replace with your machine learning model import

    app = Flask(__name__)

    # Load your pre-trained machine learning model here
    model = your_ml_model.load_model()  # Replace with your model loading logic

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        if request.method == "POST":
            # Get user input from the form
            input_data = request.form.get("input")

            # Preprocess the input data for your model (if needed)
            # ... your data preprocessing code here ...

            # Make prediction using your model
            prediction = model.predict([preprocessed_data])  # Assuming a list input

            # Format the prediction for display
            predicted_class = prediction[0]  # Assuming single class output

            return render_template("result.html", prediction=predicted_class)

        else:
            return "Something went wrong. Please try again."

    if __name__ == "__main__":
        app.run(debug=True)
    ```
5. Create HTML templates
    - The above sample `app.py` would require two HTML files `index.html` and `results.html`. These files should be located in the `templates` directory within your Flask application's root directory.
    - Remember that once the machine learning task is complete, return the results to the HTML template and update the HTML template to display the results in a user-friendly format.
    - Below is an example `index.html` file
    ```HTML
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Machine Learning Classification Project</title>
    </head>
    <body>
        <h1>Machine Learning Classification Project</h1>
        <form method="POST" action="/predict">
            <label for="input">Enter data to classify:</label>
            <input type="text" name="input" id="input" required>
            <br>
            <button type="submit">Predict</button>
        </form>
    </body>
    </html>
    ```
6. Run the application
   - In your terminal within the project directory, execute: `python3 app.py`
   - Open your web browser and navigate to `http://localhost:5000`. You should see the HTML form. 