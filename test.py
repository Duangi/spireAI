import tensorflow as tf

# 定义输入层，分别为 [7,] 和 [10,25] 的张量
input1 = tf.keras.layers.Input(shape=(7,))
input2 = tf.keras.layers.Input(shape=(10, 25))

# 对第一个输入进行特征提取
x1 = tf.keras.layers.Dense(64, activation='relu')(input1)
x1 = tf.keras.layers.Flatten()(x1)

# 对第二个输入进行特征提取
x2 = tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu')(input2)
x2 = tf.keras.layers.Flatten()(x2)

# 合并两个输入的特征
combined = tf.keras.layers.concatenate([x1, x2])

# 第一个输出头：命令选择（10类命令）
command_output = tf.keras.layers.Dense(10, activation='softmax')(combined)

# 第二个输出头：第一个数字（可选，可能为三位数字）
num1_output = tf.keras.layers.Dense(3, activation='relu')(combined)  # 生成个位、十位、百位

# 第三个输出头：第二个数字（可选）
num2_output = tf.keras.layers.Dense(3, activation='relu')(combined)

# 定义模型，包含三个输出
model = tf.keras.models.Model(inputs=[input1, input2], outputs=[command_output, num1_output, num2_output])

# 编译模型
model.compile(optimizer='adam', loss=['categorical_crossentropy', 'mse', 'mse'])
