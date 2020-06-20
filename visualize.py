from torchvision.utils import make_grid
import numpy as np

def plot_bounding_box_corners(image, coordinates, fill='red', label=None):
    # Receives as input PIL Image and list of corrdinates [x, y, x2, y2]
    # Output image with bounding box
    
    image = tf.ToPILImage()(image.squeeze())
    for coord in coordinates:
        print(coord)
        x = coord[0]
        y = coord[1]
        x2 = coord[2]
        y2 = coord[3]
        img2 = image.copy()
        draw = ImageDraw.Draw(img2)
        draw.line([x, y, x2, y], fill=fill, width=10)
        draw.line([x2, y, x2, y2], fill=fill, width=10)
        draw.line([x2, y2, x, y2], fill=fill, width=10)
        draw.line([x, y, x, y2], fill=fill, width=10)
        image = Image.blend(img2, image, 0.5)

    return image

def grid_boxes_loader(data_loader, num, nrow):

    ims = next(iter(data_loader))
    images, labels = ims
    num = min(len(images), num)
    toplot = toplot = [tf.ToTensor()(plot_bounding_box_corners(images[i],labels[i]['boxes'], fill='green')) for i in range(num)]

    return make_grid(toplot, nrow=nrow)

def grid_boxes_model(data_loader, model, num, nrow, thresh=0.5):

    ims = next(iter(data_loader))
    images, labels = ims
    num = min(len(images), num)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images = list(image.to(device) for image in images)
    labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
    exp.model.eval()
    exp.model.to(device)
    out_dict = exp.model(images)

    aux = []
    for lab in out_dict:
        aux2 = list(filter(lambda x: x[1]!=0 and x[2] > thresh, zip(lab['boxes'], lab['labels'], lab['scores'])))
        aux.append([t[0] for t in aux2])
    
    images = list(image.to('cpu') for image in images)
    toplot = [tf.ToTensor()(plot_bounding_box_corners(images[i],aux[i], fill='red')) for i in range(num)]
    toplot = [tf.ToTensor()(plot_bounding_box_corners(toplot[i],labels[i]['boxes'], fill='green')) for i in range(num)]

    return make_grid(toplot, nrow=nrow)

def grid_boxes_best(data_loader, model, num, nrow):
    ims = next(iter(data_loader))
    images, labels = ims
    num = min(len(images), num)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    images = list(image.to(device) for image in images)
    labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
    exp.model.eval()
    exp.model.to(device)
    out_dict = exp.model(images)
    aux = []
    for lab in out_dict:
        boxes = lab['boxes'].cpu().detach().numpy()
        scores = lab['scores'].cpu().detach().numpy()
        print(scores)
        if len(scores) == 0:
            print('No box')
            aux2 = []
        else:
            aux2 = [boxes[np.argmax(scores)]]
        aux.append(aux2)
    
    images = list(image.to('cpu') for image in images)
    toplot = [tf.ToTensor()(plot_bounding_box_corners(images[i],labels[i]['boxes'], fill='green')) for i in range(num)]
    toplot = [tf.ToTensor()(plot_bounding_box_corners(toplot[i],aux[i], fill='red')) for i in range(num)]

    return make_grid(toplot, nrow=nrow)

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
