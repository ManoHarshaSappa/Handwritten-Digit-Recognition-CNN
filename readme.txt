Here's a complete and professional `README.md` for your GitHub repo **`Handwritten-Digit-Recognition-CNN`**:

---

```markdown
# Handwritten Digit Recognition using CNN 🧠🔢

This project builds and trains a Convolutional Neural Network (CNN) to classify handwritten digits using the **MNIST dataset**. The model achieves over **99% accuracy** using the Adam optimizer and demonstrates strong generalization on unseen data.

## 📌 Overview

- 🔍 Dataset: MNIST (60,000 training, 10,000 testing images of digits 0–9)
- 🧠 Model: Custom CNN built using TensorFlow/Keras
- ⚙️ Optimizers: Adam and SGD (comparative analysis)
- 📊 Evaluation: Accuracy curves, loss plots, confusion matrix, classification report
- 🔎 Custom digit prediction supported

## 🧪 Technologies Used

- Python
- TensorFlow & Keras
- NumPy, Matplotlib, Seaborn
- Scikit-learn
- OpenCV (for custom image digit prediction)

## 🧱 Model Architecture

```

Conv2D(32) → MaxPooling2D
Conv2D(64) → MaxPooling2D
Flatten → Dense(128, ReLU)
Output Layer → Dense(10, Softmax)

````

## 🚀 Training Results

| Optimizer | Accuracy | Loss     |
|-----------|----------|----------|
| Adam      | ✅ **99%**     | Low loss, fast convergence |
| SGD       | ✅ **97%**     | Slower, but stable accuracy |

Visualizations include training vs. validation accuracy/loss and confusion matrices.

## 🖼️ Sample Prediction

You can predict a custom digit image:
```python
predict_single_digit('path_to_image.png', model)
````

> Make sure the image is a grayscale, 28x28 pixel digit on a white background.

## 📈 Future Improvements

* Add Dropout/L2 regularization
* Use data augmentation (rotation, scaling)
* Experiment with deeper CNNs
* Extend to multi-digit sequence recognition

## 👨‍🏫 Authors

* Saivarun T.R.
* Mohammed Tareq
* Suraj Poldas
* Mano Harsha Sappa

*Completed under the guidance of Dr. Lei Yang at George Mason University (AIT 736 - Spring 2025)*

## 📄 License

This project is licensed under the [MIT License](LICENSE) — feel free to use, modify, and build upon it!

---

```

Would you like me to generate a `LICENSE` file as well (MIT format)?
```
