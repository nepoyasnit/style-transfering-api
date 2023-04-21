import torch.nn as nn
from image_processing.normalize import Normalization
from loss_counters.style_loss import StyleLoss
from loss_counters.content_loss import ContentLoss
import logging
import torch
import torch.optim as optim


# Основной класс, перенос стиля.
class StyleTransfer:
    def __init__(self, num_steps, device='cpu', style_weight=100000, content_weight=1):
        self.num_steps = num_steps
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.device = device

        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    def get_style_model_and_losses(self, style_img, content_img):

        # Загружаем наши сохранённые 11 слоёв от VGG19 и используем как базу для создания сетки.
        cnn = torch.load('./load_model/vgg19_11_layers.pth', map_location=self.device)
        cnn = cnn.to(self.device).eval()

        normalization = Normalization(self.device).to(self.device)

        content_losses = []
        style_losses = []

        # Начинаем с нормализации
        model = nn.Sequential(normalization)

        # В цикле переименовываем слои.
        # Заменяем ReLU слои, на версию с inplace=False (иначе всё падает)
        i = 0
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)

            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)

            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                layer = nn.ReLU(inplace=False)

            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)

            model.add_module(name, layer)

            # Добавляем к указанным слоям контента и стиля - модули лоссов.
            if name in self.content_layers:
                target = model(content_img).detach()
                content_loss = ContentLoss(target)
                model.add_module("content_loss_{}".format(i), content_loss)
                content_losses.append(content_loss)

            if name in self.style_layers:
                target_feature = model(style_img).detach()
                style_loss = StyleLoss(target_feature)
                model.add_module("style_loss_{}".format(i), style_loss)
                style_losses.append(style_loss)

        return model, style_losses, content_losses

    # Создаём оптимизатор, инициализируем исходным изображением.
    @staticmethod
    def get_input_optimizer(input_img):
        optimizer = optim.LBFGS([input_img.requires_grad_()])
        return optimizer

    # Непосредственно сам метод реализующий перенос стиля
    def transfer_style(self, style_img, content_img):
        input_img = content_img.clone()
        model, style_losses, content_losses = self.get_style_model_and_losses(
            style_img, content_img)
        optimizer = self.get_input_optimizer(input_img)

        run = [0]
        while run[0] <= self.num_steps:

            def closure():
                input_img.data.clamp_(0, 1)

                optimizer.zero_grad()

                model(input_img)

                style_score = 0
                content_score = 0

                for sl in style_losses:
                    style_score += sl.loss
                for cl in content_losses:
                    content_score += cl.loss

                style_score *= self.style_weight
                content_score *= self.content_weight
                loss = style_score + content_score
                loss.backward()

                if run[0] % 50 == 0:
                    logging.info(f"run: {run[0]}")
                run[0] += 1

                return style_score + content_score

            optimizer.step(closure)

        input_img.data.clamp_(0, 1)

        return input_img
