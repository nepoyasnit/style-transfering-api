from style_transfering.transfering import StyleTransfer
import torch
from image_processing.image_processing import ImageProcessing
from PIL import Image


def run_nst(style_image, content_image):
    # Определяем доступный девайс
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Определяем максимальный размер масштабирования.
    # Если доступна gpu - 512х512, если нет - 256х256
    RESCALE_SIZE = 512 if torch.cuda.is_available() else 256

    # Создаём обработчики для изображений
    style_processing = ImageProcessing(new_size=RESCALE_SIZE, device=device)
    content_processing = ImageProcessing(new_size=RESCALE_SIZE, device=device)

    # Препроцессим изображения
    style_image = style_processing.image_loader(style_image)
    content_image = content_processing.image_loader(content_image)

    # Создаем экземпляр класса StyleTransfer и запускаем сам перенос стиля
    # После чего возвращаем изображению исходный размер и сохраняем в ByteIO
    transfer = StyleTransfer(num_steps=200, device=device)
    output_image = transfer.transfer_style(style_image, content_image)
    output_image = content_processing.get_image(output_image)

    return output_image


style_image = '/home/maksim/Pictures/studak.jpg'
input_image = '/home/maksim/Pictures/I.jpeg'
result = run_nst(style_image=style_image, content_image=input_image)
result = result.save('./static/result.jpg')
