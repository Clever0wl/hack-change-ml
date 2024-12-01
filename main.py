''' Обработка данных
import pandas as pd

# Загрузка данных
piers_path = "Piers.csv"
file_path = "TableTime.csv"

piers_data = pd.read_csv(piers_path)
data = pd.read_csv(file_path)

# Преобразование строковых дат в формат datetime
data['Дата начала действия записи расписания '] = pd.to_datetime(data['Дата начала действия записи расписания '], errors='coerce')
data['Дата окончания действия записи расписания'] = pd.to_datetime(data['Дата окончания действия записи расписания'], errors='coerce')

# Проверка минимальных и максимальных дат
print(f"Минимальная дата начала: {data['Дата начала действия записи расписания '].min()}")
print(f"Максимальная дата окончания: {data['Дата окончания действия записи расписания'].max()}")

# Очистка данных: удаление строк, где даты начала или окончания расписания отсутствуют
data = data[data['Дата начала действия записи расписания '].notna()]
data = data[data['Дата окончания действия записи расписания'].notna()]

# Проверка данных после очистки
print(f"Минимальная дата начала после очистки: {data['Дата начала действия записи расписания '].min()}")
print(f"Максимальная дата окончания после очистки: {data['Дата окончания действия записи расписания'].max()}")

# Фильтрация данных по диапазону дат: оставляем записи, которые находятся в пределах действующего расписания
current_date = pd.to_datetime("2024-10-01")  # Например, текущая дата для фильтрации
data_filtered = data[(data['Дата начала действия записи расписания '] <= current_date) &
                     (data['Дата окончания действия записи расписания'] >= current_date)]

# Проверка результатов фильтрации
print(f"Количество записей после фильтрации: {data_filtered.shape[0]}")

# Сохранение очищенного и отфильтрованного датасета
data_filtered.to_csv("filtered_schedule_data.csv", index=False)

print("Данные успешно очищены и отфильтрованы.")
'''

def load_model(onnx_model_path):
    # Загрузка модели
    return ort.InferenceSession(onnx_model_path)

def predict_trip_duration(model, input_data):
    # Преобразуем входные данные
    input_data = np.array(input_data).astype(np.float32).reshape(1, -1)
    
    # Получаем имя входного тензора
    input_name = model.get_inputs()[0].name
    
    # Выполняем предсказание
    prediction = model.run(None, {input_name: input_data})
    
    return prediction[0][0]

def get_user_input():
    # Получаем ввод от пользователя
    try:
        approach_time = float(input("Введите время подхода судна в минутах от начала суток: "))
        departure_time = float(input("Введите время отхода судна в минутах от начала суток: "))
        return [approach_time, departure_time]
    except ValueError:
        print("Ошибка: введите числовые значения для времени.")
        return None

def main():
    # Загружаем модель
    model_path = 'water_taxi_model.onnx'
    model = load_model(model_path)
    
    # Получаем данные от пользователя
    input_data = get_user_input()
    
    if input_data:
        # Получаем предсказание
        predicted_duration = predict_trip_duration(model, input_data)
        print(f"Предсказанное время поездки: {predicted_duration:.2f} минут")

if __name__ == "__main__":
    main()
