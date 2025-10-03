import json
from pathlib import Path

#  Данные для обучающей выборки (train) 
# Для каждого из 8 путей создано по 2-3 примера.
train_data = [
    # Path: Information -> ProductInquiry -> CheckPrice
    {"text": "Сколько стоит последняя модель?", "labels": [["Information", "ProductInquiry", "CheckPrice"]]},
    {"text": "Какая цена у этого товара?", "labels": [["Information", "ProductInquiry", "CheckPrice"]]},
    {"text": "Назовите стоимость, пожалуйста.", "labels": [["Information", "ProductInquiry", "CheckPrice"]]},
    
    # Path: Information -> ProductInquiry -> GetFeatures
    {"text": "Какие характеристики у этого ноутбука?", "labels": [["Information", "ProductInquiry", "GetFeatures"]]},
    {"text": "Расскажи подробнее о функциях камеры.", "labels": [["Information", "ProductInquiry", "GetFeatures"]]},
    
    # Path: Information -> CompanyInquiry -> FindContact
    {"text": "Как с вами связаться?", "labels": [["Information", "CompanyInquiry", "FindContact"]]},
    {"text": "Дайте ваш номер телефона для поддержки.", "labels": [["Information", "CompanyInquiry", "FindContact"]]},
    
    # Path: Information -> CompanyInquiry -> CheckShipping
    {"text": "Сколько времени занимает доставка?", "labels": [["Information", "CompanyInquiry", "CheckShipping"]]},
    {"text": "Вы доставляете в мой город?", "labels": [["Information", "CompanyInquiry", "CheckShipping"]]},

    # Path: Action -> AccountManagement -> UpdateProfile
    {"text": "Хочу изменить свой адрес.", "labels": [["Action", "AccountManagement", "UpdateProfile"]]},
    {"text": "Нужно обновить номер телефона в профиле.", "labels": [["Action", "AccountManagement", "UpdateProfile"]]},
    
    # Path: Action -> AccountManagement -> ResetPassword
    {"text": "Я забыл свой пароль.", "labels": [["Action", "AccountManagement", "ResetPassword"]]},
    {"text": "Помогите сбросить пароль от учетной записи.", "labels": [["Action", "AccountManagement", "ResetPassword"]]},
    
    # Path: Action -> Purchase -> AddToCart
    {"text": "Добавь этот товар в корзину.", "labels": [["Action", "Purchase", "AddToCart"]]},
    {"text": "Положи это в мою корзину, пожалуйста.", "labels": [["Action", "Purchase", "AddToCart"]]},
    
    # Path: Action -> Purchase -> CheckOrderStatus
    {"text": "Где мой заказ?", "labels": [["Action", "Purchase", "CheckOrderStatus"]]},
    {"text": "Какой статус у последней покупки?", "labels": [["Action", "Purchase", "CheckOrderStatus"]]},
    {"text": "Отследить посылку по номеру.", "labels": [["Action", "Purchase", "CheckOrderStatus"]]},
]


#  Данные для тестовой выборки (test) 
# Для каждого из 8 путей создан 1 пример.
test_data = [
    # Path: Information -> ProductInquiry -> CheckPrice
    {"text": "Какова итоговая цена?", "labels": [["Information", "ProductInquiry", "CheckPrice"]]},
    
    # Path: Information -> ProductInquiry -> GetFeatures
    {"text": "Есть ли у него водонепроницаемость?", "labels": [["Information", "ProductInquiry", "GetFeatures"]]},
    
    # Path: Information -> CompanyInquiry -> FindContact
    {"text": "Какой у вас адрес электронной почты?", "labels": [["Information", "CompanyInquiry", "FindContact"]]},
    
    # Path: Information -> CompanyInquiry -> CheckShipping
    {"text": "Какова стоимость доставки?", "labels": [["Information", "CompanyInquiry", "CheckShipping"]]},

    # Path: Action -> AccountManagement -> UpdateProfile
    {"text": "Как поменять фамилию в аккаунте?", "labels": [["Action", "AccountManagement", "UpdateProfile"]]},
    
    # Path: Action -> AccountManagement -> ResetPassword
    {"text": "Не могу войти, нужен новый пароль.", "labels": [["Action", "AccountManagement", "ResetPassword"]]},
    
    # Path: Action -> Purchase -> AddToCart
    {"text": "Я хочу купить это.", "labels": [["Action", "Purchase", "AddToCart"]]},
    
    # Path: Action -> Purchase -> CheckOrderStatus
    {"text": "Когда прибудет моя посылка?", "labels": [["Action", "Purchase", "CheckOrderStatus"]]},
]

def save_to_json(data: list, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[+] Данные успешно сохранены в: {output_path}")

def main():
    output_dir = Path("unified_datasets") / "custom_intents"
    
    save_to_json(train_data, output_dir / "train.json")    
    save_to_json(test_data, output_dir / "test.json")


if __name__ == "__main__":
    main()