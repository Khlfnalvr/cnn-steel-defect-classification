"""Arsitektur CNN: LeNet-5 (baseline) dan Mini-VGG (struktur VGGNet disederhanakan)."""
from tensorflow import keras
from tensorflow.keras import layers


def augmentation(seed=42):
    # Aktif hanya saat training. Cacat permukaan tidak bergantung orientasi,
    # jadi flip horizontal/vertikal dan geser kecil aman dipakai.
    return keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical", seed=seed),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=seed),
        layers.RandomContrast(0.1, seed=seed),
    ], name="augmentasi")


def lenet5(input_size=32, n_classes=6, augment=True):
    """LeNet-5 (LeCun et al., 1998): C1-S2-C3-S4-C5-F6-Output.
    Aktivasi ReLU (pengganti tanh) dan subsampling = average pooling."""
    inp = keras.Input((input_size, input_size, 1), name="input")
    x = augmentation()(inp) if augment else inp
    x = layers.Conv2D(6, 5, activation="relu", name="C1_conv")(x)        # 28x28x6
    x = layers.AveragePooling2D(2, name="S2_pool")(x)                    # 14x14x6
    x = layers.Conv2D(16, 5, activation="relu", name="C3_conv")(x)       # 10x10x16
    x = layers.AveragePooling2D(2, name="S4_pool")(x)                    # 5x5x16
    x = layers.Conv2D(120, 5, activation="relu", name="C5_conv")(x)      # 1x1x120
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(84, activation="relu", name="F6_fc")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="LeNet5")


def _vgg_block(x, filters, n_conv, name):
    for i in range(n_conv):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"{name}_conv{i+1}")(x)
        x = layers.BatchNormalization(name=f"{name}_bn{i+1}")(x)
        x = layers.ReLU(name=f"{name}_relu{i+1}")(x)
    return layers.MaxPooling2D(2, name=f"{name}_pool")(x)


def mini_vgg(input_size=64, n_classes=6, augment=True):
    """Mini-VGG: tumpukan konvolusi 3x3 + max pooling ala VGGNet
    (Simonyan & Zisserman, 2014), dengan BatchNorm, Global Average Pooling
    dan Dropout agar cocok untuk dataset kecil."""
    inp = keras.Input((input_size, input_size, 1), name="input")
    x = augmentation()(inp) if augment else inp
    x = _vgg_block(x, 32, 2, "block1")    # 32x32x32
    x = _vgg_block(x, 64, 2, "block2")    # 16x16x64
    x = _vgg_block(x, 128, 2, "block3")   # 8x8x128
    x = _vgg_block(x, 256, 1, "block4")   # 4x4x256
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.5, name="dropout")(x)
    x = layers.Dense(128, activation="relu", name="fc")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="MiniVGG")


@keras.utils.register_keras_serializable(package="neu")
class LRN(layers.Layer):
    """Local Response Normalization antarkanal seperti AlexNet (n=5, k=2, alpha=1e-4, beta=0.75)."""

    def __init__(self, depth_radius=2, bias=2.0, alpha=1e-4, beta=0.75, **kw):
        super().__init__(**kw)
        self.depth_radius, self.bias, self.alpha, self.beta = depth_radius, bias, alpha, beta

    def call(self, x):
        import tensorflow as tf
        return tf.nn.local_response_normalization(x, self.depth_radius, self.bias, self.alpha, self.beta)

    def get_config(self):
        return {**super().get_config(), "depth_radius": self.depth_radius, "bias": self.bias,
                "alpha": self.alpha, "beta": self.beta}


def alexnet(input_size=128, n_classes=6, augment=True):
    """AlexNet (Krizhevsky et al., 2012) yang diskalakan untuk citra grayscale 128x128:
    5 konvolusi (11x11/4, 5x5, 3x3 x3), ReLU, LRN, max pooling 3x3 stride 2 (tumpang tindih),
    dua Dense 1024 dengan Dropout 0,5. Jumlah filter dan neuron dikurangi dari 96/256/384/384/256
    dan 4096 agar sesuai dengan dataset kecil."""
    inp = keras.Input((input_size, input_size, 1), name="input")
    x = augmentation()(inp) if augment else inp
    x = layers.Conv2D(64, 11, strides=4, activation="relu", name="conv1")(x)            # 30x30x64
    x = LRN(name="lrn1")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool1")(x)                              # 14x14x64
    x = layers.Conv2D(192, 5, padding="same", activation="relu", name="conv2")(x)       # 14x14x192
    x = LRN(name="lrn2")(x)
    x = layers.MaxPooling2D(3, strides=2, name="pool2")(x)                              # 6x6x192
    x = layers.Conv2D(384, 3, padding="same", activation="relu", name="conv3")(x)       # 6x6x384
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv4")(x)       # 6x6x256
    x = layers.Conv2D(256, 3, padding="same", activation="relu", name="conv5")(x)       # 6x6x256
    x = layers.MaxPooling2D(3, strides=2, name="pool5")(x)                              # 2x2x256
    x = layers.Flatten(name="flatten")(x)                                               # 1024
    x = layers.Dropout(0.5, name="drop1")(x)
    x = layers.Dense(1024, activation="relu", name="fc1")(x)
    x = layers.Dropout(0.5, name="drop2")(x)
    x = layers.Dense(1024, activation="relu", name="fc2")(x)
    out = layers.Dense(n_classes, activation="softmax", name="output")(x)
    return keras.Model(inp, out, name="AlexNet")


MODELS = {"lenet5": (lenet5, 32), "alexnet": (alexnet, 128), "minivgg": (mini_vgg, 64)}
# learning rate awal Adam per model (AlexNet tanpa BatchNorm lebih stabil dengan lr lebih kecil)
DEFAULT_LR = {"lenet5": 1e-3, "alexnet": 3e-4, "minivgg": 1e-3}
ALIASES = {"vgg": "minivgg", "vggnet": "minivgg", "mini_vgg": "minivgg", "mini-vgg": "minivgg",
           "lenet": "lenet5", "lenet-5": "lenet5"}
