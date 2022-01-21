from Classes import *

from MyBot import *

from Menu import Menu

menu_items=["Об авторе", "О программе", "Обучение бота", "Получение рекомендации", "Выход"]
menu_title="Меню"

my_menu=Menu(menu_title, menu_items)

choice=0
while choice!=5:
    choice = my_menu.get_user_choice()
    if choice == 1:
        print("Рогальский Кирилл Вадимович, студент 1 курса магистратуры Искусственный интеллект")
    if choice == 2:
        print("Зачетное проектное задание по дисциплине Проектирование и разработка программного обеспечения, 2 семестр")
        pass
    if choice==3:
        shop = my_shop("myShop.xml")
        shop.add_sample_data(20)
        shop.add_sample_orders(1000)
        df = shop.getTrainingData()
        # Создаем бота
        bot = MYBOT(shop)
        # обучаем бота
        bot.botTraining(1)

    if choice==4:
        # получаем данные от пользователя
        sd = bot.getUserChoice()
        # строим рекомендацию и выводим рекомендованный товар
        print("Ваш рекомендованный товар: ", bot.getPrecigion(sd))
