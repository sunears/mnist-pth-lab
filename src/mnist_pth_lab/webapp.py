import io
import os
import base64
from PIL import Image, ImageOps
from flask import Flask, render_template, request, jsonify
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from mnist_pth_lab.model import build_model
from mnist_pth_lab.utils import load_model, get_logger


app = Flask(__name__, template_folder=os.path.join(os.path.dirname(__file__), 'templates'), static_folder=os.path.join(os.path.dirname(__file__), 'static'))
logger = get_logger('WebApp')

# Path to default model
DEFAULT_MODEL_PATH = os.path.join(os.getcwd(), 'models', 'mnist_cnn.pth')


def preprocess_pil_image(pil_img):
    # Convert to grayscale, invert (canvas is black on white sometimes), resize to 28x28
    img = pil_img.convert('L')
    img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.BILINEAR)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
    ])
    return transform(img).unsqueeze(0)  # shape (1,1,28,28)


def tensor_to_base64_img(tensor):
    # tensor expected HxW or CxHxW
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)
    arr = tensor.detach().cpu().numpy()
    # normalize to 0-255
    arr = arr - arr.min()
    if arr.max() != 0:
        arr = arr / arr.max()
    arr = (arr * 255).astype('uint8')
    if arr.ndim == 2:
        mode = 'L'
    else:
        mode = 'RGB'
    img = Image.fromarray(arr, mode=mode)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('ascii')


def load_default_model(device='cpu'):
    model = build_model()
    if os.path.exists(DEFAULT_MODEL_PATH):
        load_model(model, DEFAULT_MODEL_PATH, device)
    else:
        logger.warning(f"默认模型路径不存在：{DEFAULT_MODEL_PATH}，使用随机初始化模型进行演示")
    model.eval()
    return model


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json.get('image')
    if not data:
        return jsonify({'error': 'no image'}), 400
    header, encoded = data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor = preprocess_pil_image(pil_img)
    device = torch.device('cpu')
    model = load_default_model(device)
    with torch.no_grad():
        out = model(tensor.to(device))
        probs = F.softmax(out, dim=1).squeeze(0).cpu().numpy().tolist()
        pred = int(torch.argmax(out, dim=1).item())
    return jsonify({'pred': pred, 'probs': probs})


@app.route('/visualize', methods=['POST'])
def visualize():
    data = request.json.get('image')
    topk = int(request.json.get('topk', 6))
    if not data:
        return jsonify({'error': 'no image'}), 400
    header, encoded = data.split(',', 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    tensor = preprocess_pil_image(pil_img)

    device = torch.device('cpu')
    model = load_default_model(device)

    x = tensor.to(device)
    # conv1 activations before pool
    with torch.no_grad():
        act1 = F.relu(model.conv1(x))  # (1, C1, H, W)
        pooled1 = model.pool(act1)
        act2 = F.relu(model.conv2(pooled1))  # (1, C2, H2, W2)

    def top_channel_images(act, topk):
        # act: (1,C,H,W)
        c = act.shape[1]
        scores = act.abs().mean(dim=(2,3)).squeeze(0)  # (C,)
        vals, idx = torch.topk(scores, min(topk, c))
        imgs = []
        for i in idx.tolist():
            amap = act[0, i:i+1, :, :]
            # upsample to 28x28
            amap_up = F.interpolate(amap.unsqueeze(0), size=(28,28), mode='bilinear', align_corners=False).squeeze(0)
            b64 = tensor_to_base64_img(amap_up.squeeze(0))
            imgs.append({'channel': i, 'image': b64, 'score': float(scores[i].item())})
        return imgs

    conv1_top = top_channel_images(act1, topk)
    conv2_top = top_channel_images(act2, topk)

    # Also provide kernel images (weights)
    def kernels_to_imgs(conv_layer, topk):
        w = conv_layer.weight.data.clone().cpu()  # (out, in, k, k)
        outc = w.shape[0]
        imgs = []
        for i in range(min(topk, outc)):
            kern = w[i]
            # if single channel, take first channel
            if kern.shape[0] == 1:
                kern_img = kern[0]
            else:
                # merge channels by mean
                kern_img = kern.mean(dim=0)
            b64 = tensor_to_base64_img(kern_img)
            imgs.append({'channel': i, 'kernel': b64})
        return imgs

    kernels1 = kernels_to_imgs(model.conv1, topk)
    kernels2 = kernels_to_imgs(model.conv2, topk)

    return jsonify({'conv1_top': conv1_top, 'conv2_top': conv2_top, 'kernels1': kernels1, 'kernels2': kernels2})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
