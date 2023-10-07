from flask import Flask, render_template, request, url_for
import os


from utilities import ConvertToSportName, PredictOneImage, predictFrames, SportsDictionary

app = Flask(__name__)


# routes
@app.route("/")
def main():
	return render_template("Home.html")

@app.route("/Image", methods = ['GET', 'POST'])
def Image():
	return render_template("Image.html")


@app.route("/submit_image", methods = ['GET', 'POST'])
def get_output_image():
	label = None
	if request.method == 'POST':
		Input = request.files['my_image']

		Input_path = "static/" + Input.filename
		Input.save(Input_path)
		
		pred = PredictOneImage (Input_path)
		label = ConvertToSportName(pred,SportsDictionary)

	return render_template("Image.html", label = label, Input_filename = Input.filename)


@app.route("/Video", methods = ['GET', 'POST'])
def Video():
	return render_template("Video.html")

@app.route("/submit_video", methods = ['GET', 'POST'])
def get_output_video():
	label = None
	if request.method == 'POST':
		Input = request.files['my_video']

		Input_path = "static/" + Input.filename
		Input.save(Input_path)
	
		label = predictFrames(Input_path, SportsDictionary)

	return render_template("Video.html", label = label, Input_filename = Input.filename)


if __name__ == '__main__':
    app.run(debug=True)