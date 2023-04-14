import PIL.Image as Image
import PIL
import torchvision.transforms as transforms
from io import BytesIO
import torch


class ImageProcessing:
    def __init__(self, new_size, device):
        self.new_size = new_size
        self.device = device
        self.image_size = None

    def image_loader(self, image_name):
        image = Image.open(image_name)
        self.image_size = image.size

        # Для сохранения соотношения сторон, ресайзим изображение добавляя паддинг.
        image = PIL.ImageOps.pad(image, (self.new_size, self.new_size))
        loader = transforms.ToTensor()
        image = loader(image).unsqueeze(0)

        return image.to(self.device, torch.float)

    def get_image(self, tensor):
        image = tensor.cpu().clone()
        image = image.squeeze(0)
        unloader = transforms.ToPILImage()
        image = unloader(image)

        # Возвращаем изображению исходный размер.
        image = PIL.ImageOps.fit(image, self.image_size)

        # Записываем изображение в буфер,
        # в таком виде его надо отправлять пользователю
        bio = BytesIO()
        bio.name = 'output.jpeg'
        image.save(bio, 'JPEG')
        bio.seek(0)

        return bio
