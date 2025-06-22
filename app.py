from flask import Flask, render_template, request

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


## Route for a home page
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint() -> str:
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("writing_score")),
            writing_score=float(request.form.get("reading_score")),
        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        predict_pipeline = PredictPipeline()

        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)

        print("After Prediction")
        return render_template("home.html", results=f"{results[0]:.2f}")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
