import os
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from tkinter import simpledialog
import requests
from io import BytesIO

# Конфигурация
CLASSES = ["Огурец", "Помидор", "Салат", "Не растение"]
BATCH_SIZE = 8
MAX_EPOCHS = 30
PATIENCE = 3
VAL_SPLIT = 0.2
IMG_SIZE = 224

# Пути к данным
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")
VAL_DIR = os.path.join(DATASET_DIR, "val")
TEST_DIR = os.path.join(DATASET_DIR, "test")
RETRAIN_DIR = "retrain_data"
MODEL_PATH = "plant_model.pth"
METRICS_PATH = "training_metrics.png"

# PlantNet API
PLANTNET_API_KEY = ""
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"

# Создаем структуру папок
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)
os.makedirs(RETRAIN_DIR, exist_ok=True)
for class_name in CLASSES :
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(TEST_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(RETRAIN_DIR, class_name), exist_ok=True)


# Загрузка/создание модели
def load_model(pretrained=True) :
    model = models.efficientnet_b0(pretrained=pretrained)
    model.classifier[1] = nn.Linear(1280, len(CLASSES))
    if os.path.exists(MODEL_PATH) :
        model.load_state_dict(torch.load(MODEL_PATH))
    return model


# Преобразования изображений
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
])


# Dataset
class PlantDataset(Dataset) :
    def __init__(self, root_dir, transform=None) :
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []

        for label, class_name in enumerate(CLASSES) :
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir) :
                continue
            for img_name in os.listdir(class_dir) :
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(label)

    def __len__(self) :
        return len(self.images)

    def __getitem__(self, idx) :
        img_path = self.images[idx]
        try :
            image = Image.open(img_path).convert("RGB")
            if self.transform :
                image = self.transform(image)
            return image, self.labels[idx]
        except :
            return None


def collate_fn(batch) :
    batch = list(filter(lambda x : x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


# Обучение модели
def train_model(initial_train=True) :
    # Определяем пути к данным
    if initial_train :
        train_dir = TRAIN_DIR
        val_dir = VAL_DIR
    else :
        train_dir = RETRAIN_DIR
        val_dir = VAL_DIR  # Можно использовать оригинальные валидационные данные

    # Проверка данных
    if not os.listdir(train_dir) :
        messagebox.showerror("Ошибка", f"Нет данных для {'обучения' if initial_train else 'дообучения'}!")
        return

    # Загрузка данных
    train_dataset = PlantDataset(train_dir, transform=train_transform)
    val_dataset = PlantDataset(val_dir, transform=val_transform)

    # Удаляем битые изображения
    train_dataset.images = [img for img, label in zip(train_dataset.images, train_dataset.labels)
                            if Image.open(img).convert("RGB") is not None]
    train_dataset.labels = [label for img, label in zip(train_dataset.images, train_dataset.labels)
                            if Image.open(img).convert("RGB") is not None]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Настройки обучения
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss' : [], 'val_loss' : [], 'val_acc' : []}

    progress = tqdm(range(MAX_EPOCHS), desc="Обучение" if initial_train else "Дообучение")
    model.train()

    for epoch in progress :
        # Фаза обучения
        train_loss = 0.0
        for images, labels in train_loader :
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Фаза валидации
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad() :
            for images, labels in val_loader :
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(accuracy)

        # Ранняя остановка
        if val_loss < best_val_loss :
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else :
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE :
                progress.set_postfix({"stop" : f"Эпоха {epoch + 1}"})
                break

        progress.set_postfix({
            "train_loss" : f"{train_loss:.4f}",
            "val_loss" : f"{val_loss:.4f}",
            "val_acc" : f"{accuracy:.2f}%"
        })

    # Визуализация результатов
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Validation Accuracy', color='green')
    plt.legend()

    plt.savefig(METRICS_PATH)
    plt.close()

    messagebox.showinfo(
        "Обучение завершено" if initial_train else "Дообучение завершено",
        f"Лучшая точность: {max(history['val_acc']):.2f}%\n"
        f"Финальный val_loss: {history['val_loss'][-1]:.4f}"
    )


# GUI приложение
class PlantClassifierApp :
    def __init__(self, root) :
        self.root = root
        self.root.title("Классификатор растений")
        self.root.geometry("900x850")

        # Переменные
        self.image_path = None
        self.correct_class = tk.IntVar(value=-1)
        self.current_test_image = None
        self.test_images = []
        self.current_test_index = 0

        # Запрос пользователю при запуске
        self.ask_model_loading()

        # Виджеты
        self.create_widgets()

    def ask_model_loading(self) :
        """Спрашивает пользователя, как загрузить модель"""
        global model

        if not os.path.exists(MODEL_PATH) :
            # Если модели нет, сразу начинаем обучение
            model = load_model(pretrained=True)
            messagebox.showinfo("Информация", "Модель не найдена. Будет выполнено обучение новой модели.")
            train_model(initial_train=True)
            return

        answer = messagebox.askyesnocancel(
            "Загрузка модели",
            "Обнаружена сохраненная модель. Хотите загрузить её?\n"
            "Да - загрузить существующую модель\n"
            "Нет - обучить новую модель\n"
            "Отмена - использовать базовую модель без обучения"
        )

        if answer is None :  # Отмена - используем базовую модель
            model = load_model(pretrained=True)
            messagebox.showinfo("Информация", "Используется базовая модель (без дообучения)")
        elif answer :  # Да - загружаем сохраненную модель
            model = load_model(pretrained=False)
            messagebox.showinfo("Информация", "Сохраненная модель успешно загружена")
        else :  # Нет - обучаем новую модель
            model = load_model(pretrained=True)
            train_model(initial_train=True)

    def create_widgets(self) :
        # Заголовок
        tk.Label(self.root, text="Классификатор растений", font=("Arial", 16, "bold")).pack(pady=10)

        # 1. Загрузка датасета
        self.create_dataset_section()

        # 2. Обучение модели
        self.create_training_section()

        # 3. Тестирование
        self.create_testing_section()

        # 4. Отображение результатов
        self.create_results_section()

    def create_dataset_section(self) :
        frame = tk.LabelFrame(self.root, text="1. Подготовка данных", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Загрузка тренировочных данных по классам
        tk.Label(frame, text="Загрузить тренировочные данные:", anchor="w").pack(fill="x")

        class_btn_frame = tk.Frame(frame)
        class_btn_frame.pack(fill="x", pady=5)

        for i, class_name in enumerate(CLASSES) :
            tk.Button(
                class_btn_frame,
                text=f"Загрузить {class_name}",
                command=lambda cn=class_name : self.load_class_images(cn, "train")
            ).pack(side="left", padx=5, expand=True)

        # Загрузка тестовых данных
        tk.Label(frame, text="Загрузить тестовые данные:", anchor="w").pack(fill="x", pady=(10, 0))
        tk.Button(frame, text="Загрузить тестовые изображения", command=self.load_test_images).pack(pady=5)

    def plantnet_identify(self, image_path) :
        """Определение растения через PlantNet API"""
        try :
            with open(image_path, 'rb') as img_file :
                img_data = img_file.read()

            files = {
                'images' : ('image.jpg', BytesIO(img_data)),
                'organs' : (None, 'leaf')  # Можно изменить на 'flower', 'fruit' и т.д.
            }

            params = {
                'api-key' : PLANTNET_API_KEY,
                'include-related-images' : 'true',
                'no-reject' : 'false',
                'lang' : 'ru'
            }

            response = requests.post(PLANTNET_URL, files=files, params=params)
            response.raise_for_status()
            data = response.json()

            if 'results' in data and data['results'] :
                return data
            return None

        except Exception as e :
            print(f"PlantNet API error: {e}")
            return None

    def create_training_section(self) :
        frame = tk.LabelFrame(self.root, text="2. Обучение модели", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)

        tk.Button(btn_frame, text="Обучить модель",
                  command=lambda : train_model(initial_train=True)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Дообучить модель",
                  command=lambda : train_model(initial_train=False)).pack(side="left", padx=5)

    def create_testing_section(self) :
        frame = tk.LabelFrame(self.root, text="3. Тестирование модели", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        # Отображение изображения
        self.canvas = tk.Canvas(frame, width=IMG_SIZE, height=IMG_SIZE, bg="gray")
        self.canvas.pack(pady=5)

        # Навигация по тестовым изображениям
        nav_frame = tk.Frame(frame)
        nav_frame.pack(fill="x", pady=5)

        self.btn_prev = tk.Button(nav_frame, text="< Назад", command=self.prev_test_image, state=tk.DISABLED)
        self.btn_prev.pack(side="left", padx=5)

        self.btn_next = tk.Button(nav_frame, text="Вперед >", command=self.next_test_image, state=tk.DISABLED)
        self.btn_next.pack(side="left", padx=5)

        self.btn_predict = tk.Button(nav_frame, text="Предсказать", command=self.run_prediction, state=tk.DISABLED)
        self.btn_predict.pack(side="left", padx=5)

        # Отображение результата
        self.result_label = tk.Label(frame, text="", font=("Arial", 12))
        self.result_label.pack(pady=5)

        # Обратная связь
        feedback_frame = tk.Frame(frame)
        feedback_frame.pack(fill="x", pady=5)

        tk.Label(feedback_frame, text="Правильный класс:").pack(side="left")

        for i, class_name in enumerate(CLASSES) :
            tk.Radiobutton(
                feedback_frame,
                text=class_name,
                variable=self.correct_class,
                value=i
            ).pack(side="left", padx=5)

        tk.Button(frame, text="Сохранить правку", command=self.save_feedback).pack(pady=5)

    def create_results_section(self) :
        frame = tk.LabelFrame(self.root, text="4. Результаты", padx=10, pady=10)
        frame.pack(fill="x", padx=10, pady=5)

        self.metrics_label = tk.Label(frame, text="Графики обучения появятся здесь после обучения")
        self.metrics_label.pack(pady=5)

        btn_frame = tk.Frame(frame)
        btn_frame.pack(fill="x", pady=5)

        tk.Button(btn_frame, text="Показать графики", command=self.show_metrics).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Сохранить модель", command=self.save_model).pack(side="left", padx=5)

    def load_class_images(self, class_name, dataset_type="train") :
        target_dir = os.path.join(DATASET_DIR, dataset_type, class_name)

        # Очищаем целевую директорию
        if os.path.exists(target_dir) :
            shutil.rmtree(target_dir)
        os.makedirs(target_dir)

        # Открываем диалог выбора файлов
        filepaths = filedialog.askopenfilenames(
            title=f"Выберите изображения {class_name}",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not filepaths :
            return

        # Копируем изображения
        count = 0
        for filepath in filepaths :
            try :
                # Проверяем, что файл является изображением
                with Image.open(filepath) as img :
                    img.verify()

                # Копируем в целевую директорию
                filename = os.path.basename(filepath)
                shutil.copy(filepath, os.path.join(target_dir, filename))
                count += 1
            except :
                continue

        messagebox.showinfo("Успех", f"Загружено {count} изображений класса {class_name}")

    def load_test_images(self) :
        # Очищаем тестовую директорию
        test_all_dir = os.path.join(TEST_DIR, "all")
        if os.path.exists(test_all_dir) :
            shutil.rmtree(test_all_dir)
        os.makedirs(test_all_dir)

        # Открываем диалог выбора файлов
        filepaths = filedialog.askopenfilenames(
            title="Выберите тестовые изображения",
            filetypes=[("Изображения", "*.jpg *.jpeg *.png *.bmp")]
        )

        if not filepaths :
            return

        # Копируем изображения
        self.test_images = []
        count = 0
        for filepath in filepaths :
            try :
                with Image.open(filepath) as img :
                    img.verify()

                filename = os.path.basename(filepath)
                dest_path = os.path.join(test_all_dir, filename)
                shutil.copy(filepath, dest_path)
                self.test_images.append(dest_path)
                count += 1
            except :
                continue

        if count > 0 :
            self.current_test_index = 0
            self.show_test_image(0)
            self.btn_next.config(state=tk.NORMAL if len(self.test_images) > 1 else tk.DISABLED)
            self.btn_prev.config(state=tk.DISABLED)
            self.btn_predict.config(state=tk.NORMAL)
            messagebox.showinfo("Успех", f"Загружено {count} тестовых изображений")
        else :
            messagebox.showwarning("Ошибка", "Не найдено подходящих изображений")

    def show_test_image(self, index) :
        if 0 <= index < len(self.test_images) :
            self.current_test_index = index
            self.current_test_image = self.test_images[index]

            image = Image.open(self.current_test_image)
            image.thumbnail((IMG_SIZE, IMG_SIZE))
            self.photo = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            # Обновляем состояние кнопок навигации
            self.btn_prev.config(state=tk.NORMAL if index > 0 else tk.DISABLED)
            self.btn_next.config(state=tk.NORMAL if index < len(self.test_images) - 1 else tk.DISABLED)

            # Очищаем предыдущий результат
            self.result_label.config(text="")
            self.correct_class.set(-1)

    def prev_test_image(self) :
        self.show_test_image(self.current_test_index - 1)

    def next_test_image(self) :
        self.show_test_image(self.current_test_index + 1)

    def run_prediction(self) :
        if not self.current_test_image :
            return

        try :
            # Предсказание нашей моделью
            image = Image.open(self.current_test_image).convert("RGB")
            image_tensor = val_transform(image).unsqueeze(0)

            with torch.no_grad() :
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                confidence = confidence.item()
                predicted_class = predicted_class.item()

            result_text = ""

            # Если уверенность низкая - используем PlantNet
            if confidence < 0.7 :
                result_text += "🔍 Наша модель не уверена. Запрос к PlantNet...\n"

                # Вызываем PlantNet API
                plantnet_data = self.plantnet_identify(self.current_test_image)

                if plantnet_data :
                    best_match = plantnet_data['results'][0]
                    scientific_name = best_match['species']['scientificNameWithoutAuthor']
                    common_name = best_match['species'].get('commonNames', ['Неизвестно'])[0]
                    score = best_match['score']

                    result_text += (
                        f"\n🌿 PlantNet определил растение как:\n"
                        f"Научное название: {scientific_name}\n"
                        f"Общее название: {common_name}\n"
                        f"Уверенность: {score * 100:.1f}%\n"
                    )

                    # Добавляем похожие изображения
                    if 'images' in best_match :
                        result_text += "\nПохожие изображения:\n"
                        for i, img in enumerate(best_match['images'][:3], 1) :
                            result_text += f"{i}. {img['url']}\n"
                else :
                    result_text += "Не удалось определить растение через PlantNet\n"

                # Все равно показываем предсказание нашей модели
                result_text += "\nНаше предсказание:\n"

            # Добавляем предсказание нашей модели
            result_text += f"{CLASSES[predicted_class]}: {confidence * 100:.1f}%\n"

            # Топ-3 варианта
            top_probs, top_classes = torch.topk(probabilities, 3)
            for i in range(top_probs.size(1)) :
                result_text += f"{CLASSES[top_classes[0][i]]}: {top_probs[0][i] * 100:.1f}%\n"

            self.result_label.config(text=result_text)

        except Exception as e :
            messagebox.showerror("Ошибка", f"Не удалось обработать изображение: {str(e)}")

    def save_feedback(self) :
        if not self.current_test_image or self.correct_class.get() == -1 :
            messagebox.showerror("Ошибка", "Сначала загрузите изображение и выберите класс!")
            return

        correct_class_idx = self.correct_class.get()
        class_name = CLASSES[correct_class_idx]
        class_dir = os.path.join(RETRAIN_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)

        img_name = os.path.basename(self.current_test_image)
        save_path = os.path.join(class_dir, img_name)
        shutil.copy(self.current_test_image, save_path)

        messagebox.showinfo("Успех", f"Изображение сохранено как {class_name} для дообучения!")
        self.correct_class.set(-1)

    def show_metrics(self) :
        if os.path.exists(METRICS_PATH) :
            image = Image.open(METRICS_PATH)
            image.show()
        else :
            messagebox.showwarning("Внимание", "Графики обучения не найдены!")

    def save_model(self) :
        torch.save(model.state_dict(), MODEL_PATH)
        messagebox.showinfo("Успех", f"Модель сохранена в {MODEL_PATH}")


# Запуск приложения
if __name__ == "__main__" :
    root = tk.Tk()
    app = PlantClassifierApp(root)
    root.mainloop()