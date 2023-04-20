import torch.nn as nn
import torch.nn.functional as F
import torch


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = self.gram_matrix(target_feature).detach()  # константа, нужно убрать из вычислительного графа
        self.loss = F.mse_loss(self.target, self.target)  # Инициализируемся исходным изображением

    # Вычисление матрицы Грама.
    # В данном случае реализовал как статический метод, так удобнее , всё зашито в единый класс.
    @staticmethod
    def gram_matrix(input):
        batch_size, feature_maps, h, w = input.size()
        # Вытягиваем в вектор каждую карту признаков
        # Батч-сайз в нашем случае всегда будет 1, потому, что передаем по 1 фото.
        features = input.view(batch_size * feature_maps, h * w)  # resise F_XL into \hat F_XL
        G = torch.mm(features, features.t())  # Вычисляем матрицу Грама.
        # Нормализуем значения в матрице Грама.
        return G.div(batch_size * feature_maps * h * w)

    def forward(self, input):
        gram = self.gram_matrix(input)
        self.loss = F.mse_loss(gram, self.target)
        return input
