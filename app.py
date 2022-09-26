import flask
from flask import Flask, request, Response
import os
import uuid
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import threading
import adapter.famnet as famnet

lock = threading.Semaphore(1)

app = flask.Flask(
    __name__,
    static_url_path="",
    static_folder="static",
)

app.config["IMAGES_FOLDER"] = "images"


@app.route('/upload', methods=['GET', "POST"])
def upload_file():
    if "image" not in request.files:
        if request.method == "GET":
            return """
            <h1>Upload new File</h1>
            <form method="post" enctype="multipart/form-data">
            <input type="file" name="image">
            <input type="submit">
            </form>
            """

        return dict(message="Invalid upload format"), 404

    image = request.files['image']
    image_id = uuid.uuid4().hex

    path = os.path.join(app.config['IMAGES_FOLDER'], image_id)
    image.save(path)

    return dict(image_id=image_id, status="ok")


@app.route('/run', methods=['GET', "POST"])
def run_model():
    if request.method == "GET":
        return """
        <h1>Upload new File</h1>
        <form method="post" enctype="multipart/form-data">
        <input type="file" name="image">
        <input type="submit">
        </form>
        """


@app.route("/demo")
def demo_web_app():
    return flask.send_from_directory("html", "demo.html")


@app.route("/plot/<filename>", methods=["POST"])
def demo_plot(filename):

    filepath = os.path.join("static/img", filename)
    raw_image = Image.open(filepath)

    # print(request.get_data())

    data = request.get_json(force=True)
    tlbr = np.array(data)

    print(tlbr)

    import torch
    from torchvision.transforms.functional import to_tensor

    image = raw_image.convert("RGB")
    image = to_tensor(image).to(device=famnet.device).unsqueeze(0)
    tlbr = torch.tensor(tlbr, device=famnet.device).unsqueeze(0)

    lock.acquire()
    heatmap = famnet.run(image, tlbr)
    lock.release()

    plt.clf()

    plt.imshow(raw_image, alpha=0.8)
    plt.imshow(heatmap, alpha=0.6)
    plt.title("Count=" + str(heatmap.sum()))

    os.remove("static/img/__temp.png")
    plt.savefig("static/img/__temp.png")

    raw_image.close()

    return flask.send_file("static/img/__temp.png", mimetype='image/png')


app.run(debug=True)
