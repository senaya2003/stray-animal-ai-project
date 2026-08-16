from flask import Flask, render_template, request
import joblib
import sqlite3
import os
import numpy as np
import pandas as pd

from werkzeug.utils import secure_filename
from PIL import Image

import tensorflow as tf


# ==================================================
# Flask Application
# ==================================================

app = Flask(__name__)


# ==================================================
# Paths
# ==================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# --------------------------------------------------
# Module 1 - Health Status Model
# --------------------------------------------------

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "health_status_model.joblib"
)


# --------------------------------------------------
# Module 2 - Image Classification Model
# --------------------------------------------------

IMAGE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "animal_image_model.keras"
)


IMAGE_CLASS_NAMES_PATH = os.path.join(
    BASE_DIR,
    "models",
    "image_class_names.npy"
)


# --------------------------------------------------
# SQLite Database
# --------------------------------------------------

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "database",
    "stray_animals.db"
)


# --------------------------------------------------
# Uploaded Images
# --------------------------------------------------

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ==================================================
# Load Module 1 Machine Learning Model
# ==================================================

try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "Module 1 model loaded successfully."
    )

except Exception as error:

    print(
        "Error loading Module 1 model:",
        error
    )

    model = None


# ==================================================
# Load Module 2 Image Classification Model
# ==================================================

image_model = None
image_class_names = []


if os.path.exists(IMAGE_MODEL_PATH):

    try:

        image_model = tf.keras.models.load_model(
            IMAGE_MODEL_PATH
        )

        print(
            "Module 2 image model loaded successfully."
        )

    except Exception as error:

        print(
            "Error loading Module 2 image model:",
            error
        )

else:

    print(
        "Module 2 image model not found."
    )

    print(
        "Expected:",
        IMAGE_MODEL_PATH
    )


# --------------------------------------------------
# Load Image Class Names
# --------------------------------------------------

if os.path.exists(
    IMAGE_CLASS_NAMES_PATH
):

    try:

        image_class_names = np.load(
            IMAGE_CLASS_NAMES_PATH,
            allow_pickle=True
        ).tolist()

        print(
            "Image classes:",
            image_class_names
        )

    except Exception as error:

        print(
            "Error loading image class names:",
            error
        )

else:

    print(
        "Image class names file not found."
    )


# ==================================================
# Database Connection
# ==================================================

def get_db_connection():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    connection.row_factory = sqlite3.Row

    return connection


# ==================================================
# Home Page - MODULE 1
# ==================================================

@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction=None
    )


# ==================================================
# MODULE 1
# Animal Health Prediction
# ==================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        # ------------------------------------------
        # Get form data
        # ------------------------------------------

        animal_type = request.form[
            "animal_type"
        ]

        breed = request.form[
            "breed"
        ]

        age = float(
            request.form[
                "age"
            ]
        )

        gender = request.form[
            "gender"
        ]

        location = request.form[
            "location"
        ]

        rescue_status = request.form[
            "rescue_status"
        ]


        # ------------------------------------------
        # Check Module 1 model
        # ------------------------------------------

        if model is None:

            return render_template(
                "index.html",
                prediction="Model could not be loaded.",
                animal_type=animal_type,
                breed=breed,
                age=age,
                gender=gender,
                location=location,
                rescue_status=rescue_status
            )


        # ------------------------------------------
        # Create input DataFrame
        # ------------------------------------------

        input_data = pd.DataFrame({

            "Animal_Type": [
                animal_type
            ],

            "Breed": [
                breed
            ],

            "Age": [
                age
            ],

            "Gender": [
                gender
            ],

            "Location": [
                location
            ],

            "Rescue_Status": [
                rescue_status
            ]

        })


        # ------------------------------------------
        # Make prediction
        # ------------------------------------------

        prediction_encoded = model.predict(
            input_data
        )[0]


        # Your trained pipeline should already
        # return the original text label.
        prediction = prediction_encoded


        # ------------------------------------------
        # Store prediction in SQLite
        # ------------------------------------------

        connection = get_db_connection()

        try:

            connection.execute(
                """
                INSERT INTO predictions
                (
                    animal_type,
                    breed,
                    age,
                    gender,
                    location,
                    rescue_status,
                    predicted_health_status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,

                (
                    animal_type,
                    breed,
                    age,
                    gender,
                    location,
                    rescue_status,
                    str(prediction)
                )
            )

            connection.commit()

        except sqlite3.Error as error:

            print(
                "Database error:",
                error
            )

        finally:

            connection.close()


        # ------------------------------------------
        # Return Module 1 page
        # ------------------------------------------

        return render_template(

            "index.html",

            prediction=prediction,

            animal_type=animal_type,

            breed=breed,

            age=age,

            gender=gender,

            location=location,

            rescue_status=rescue_status
        )


    except ValueError:

        return render_template(
            "index.html",
            prediction="Please enter a valid age."
        )


    except Exception as error:

        print(
            "Prediction error:",
            error
        )

        return render_template(
            "index.html",
            prediction="An error occurred while making the prediction."
        )


# ==================================================
# MODULE 2
# Image Identification Page
# ==================================================

@app.route("/image")
def image_page():

    return render_template(

        "image_prediction.html",

        prediction=None,

        confidence=None,

        image_path=None
    )


# ==================================================
# MODULE 2
# Image Health Prediction
# ==================================================

@app.route(
    "/image-predict",
    methods=["POST"]
)
def image_predict():

    # ----------------------------------------------
    # Check whether image was uploaded
    # ----------------------------------------------

    if "animal_image" not in request.files:

        return render_template(

            "image_prediction.html",

            prediction="No image uploaded.",

            confidence=None,

            image_path=None
        )


    image_file = request.files[
        "animal_image"
    ]


    # ----------------------------------------------
    # Check filename
    # ----------------------------------------------

    if image_file.filename == "":

        return render_template(

            "image_prediction.html",

            prediction="Please select an image.",

            confidence=None,

            image_path=None
        )


    # ----------------------------------------------
    # Check Module 2 model
    # ----------------------------------------------

    if image_model is None:

        return render_template(

            "image_prediction.html",

            prediction="Image classification model is not available.",

            confidence=None,

            image_path=None
        )


    # ----------------------------------------------
    # Create safe filename
    # ----------------------------------------------

    filename = secure_filename(
        image_file.filename
    )


    # ----------------------------------------------
    # Save uploaded image
    # ----------------------------------------------

    image_path = os.path.join(

        app.config[
            "UPLOAD_FOLDER"
        ],

        filename
    )


    image_file.save(
        image_path
    )


    # ----------------------------------------------
    # Open image
    # ----------------------------------------------

    try:

        image = Image.open(
            image_path
        ).convert("RGB")

    except Exception as error:

        print(
            "Image loading error:",
            error
        )

        return render_template(

            "image_prediction.html",

            prediction="Invalid image file.",

            confidence=None,

            image_path=None
        )


    # ----------------------------------------------
    # Resize image
    # ----------------------------------------------

    image = image.resize(
        (224, 224)
    )


    # ----------------------------------------------
    # Convert image to NumPy array
    # ----------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )


    # ----------------------------------------------
    # Add batch dimension
    # ----------------------------------------------

    image_array = np.expand_dims(
        image_array,
        axis=0
    )


    # ----------------------------------------------
    # Make prediction
    # ----------------------------------------------

    try:

        predictions = image_model.predict(
            image_array,
            verbose=0
        )

    except Exception as error:

        print(
            "Image prediction error:",
            error
        )

        return render_template(

            "image_prediction.html",

            prediction="Unable to process this image.",

            confidence=None,

            image_path=None
        )


    # ----------------------------------------------
    # Find predicted class
    # ----------------------------------------------

    predicted_index = int(
        np.argmax(
            predictions[0]
        )
    )


    # ----------------------------------------------
    # Get predicted label
    # ----------------------------------------------

    if image_class_names:

        prediction = image_class_names[
            predicted_index
        ]

    else:

        prediction = str(
            predicted_index
        )


    # ----------------------------------------------
    # Calculate confidence
    # ----------------------------------------------

    confidence = float(
        predictions[0][
            predicted_index
        ]
    ) * 100


    # ----------------------------------------------
    # Create browser image URL
    # ----------------------------------------------

    image_url = (
        "/static/uploads/"
        + filename
    )


    # ----------------------------------------------
    # Store Module 2 prediction
    # ----------------------------------------------

    connection = get_db_connection()

    try:

        connection.execute(
            """
            INSERT INTO image_predictions
            (
                image_filename,
                predicted_health_status,
                confidence
            )
            VALUES (?, ?, ?)
            """,

            (
                filename,
                str(prediction),
                confidence
            )
        )

        connection.commit()

    except sqlite3.Error as error:

        print(
            "Image database error:",
            error
        )

    finally:

        connection.close()


    # ----------------------------------------------
    # Display result
    # ----------------------------------------------

    return render_template(

        "image_prediction.html",

        prediction=prediction,

        confidence=round(
            confidence,
            2
        ),

        image_path=image_url
    )


# ==================================================
# Run Application
# ==================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000
    )