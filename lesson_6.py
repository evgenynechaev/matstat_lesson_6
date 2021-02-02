import numpy as np
import pandas as pd
from scipy.stats import norm, t, pearsonr


def my_mean(in_array: list):
    return sum(in_array) / len(in_array)


def my_var_shifted(in_array: list):
    mean = my_mean(in_array)
    return sum(map(lambda x: (x - mean) ** 2, in_array)) / len(in_array)


def my_var_not_shifted(in_array: list):
    mean = my_mean(in_array)
    return sum(map(lambda x: (x - mean) ** 2, in_array)) / (len(in_array) - 1)


def cov_not_shifted(in_array_1: list, in_array_2: list):
    mean_1 = my_mean(in_array_1)
    mean_2 = my_mean(in_array_2)
    n = len(in_array_1)
    total = 0
    for i in range(n):
        total += (in_array_1[i] - mean_1) * (in_array_2[i] - mean_2)
    return total / (n - 1)


def task_1():
    """
    Задача 1:
        Даны значения величины заработной платы заемщиков банка (zp) и
        значения их поведенческого кредитного скоринга (ks):
        zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
        ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832].
        Найдите ковариацию этих двух величин с помощью элементарных действий, а затем с помощью функции cov из numpy
        Полученные значения должны быть равны.
        Найдите коэффициент корреляции Пирсона с помощью ковариации и среднеквадратичных отклонений двух признаков,
        а затем с использованием функций из библиотек numpy и pandas.
    Решение:
        n = 10
        Ковариация_не_смещ = (1/(n-1)) * sum((x_i - x_mean) * (e_i - y_mean))
        коэффициент корреляции Пирсона r = cov_не_смещ / (sigma_x_не_смещ * sigma_y_не_смещ)
    Ответ:
        cov = 10175.3778
        r = 0.8875
    """
    print("Задача 1")
    zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110]
    ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]
    p = np.array(zp)
    s = np.array(ks)
    cov_not_shift = cov_not_shifted(zp, ks)
    sigma_zp = my_var_not_shifted(zp) ** 0.5
    sigma_ks = my_var_not_shifted(ks) ** 0.5
    print(f"    Расчет собственными функциями:")
    print(f"        cov не смещ = {cov_not_shift:.4f}")
    print(f"        находим коэффициент корреляции Пирсона:")
    r = cov_not_shift / (sigma_zp * sigma_ks)
    print(f"        r = {r:.4f}")
    print()
    print(f"    Проверка при помощи библиотеки numpy:")
    cov_np = np.cov(p, s, ddof=1)[0][1]
    sigma_zp_np = np.std(p, ddof=1)
    sigma_ks_np = np.std(s, ddof=1)
    r_np = cov_np / (sigma_zp_np * sigma_ks_np)
    corrcoef_np = np.corrcoef(p, s)[0][1]
    print(f"        numpy cov = {cov_np:.4f}")
    print(f"        numpy r = {r_np:.4f}")
    print(f"        numpy corrcoef = {corrcoef_np:.4f}")
    print()
    print(f"    Проверка при помощи функции pearsonr:")
    r, _ = pearsonr(zp, ks)
    print(f"        pearson r = {r:.4f}")
    print()
    print(f"    Проверка при помощи библиотеки pandas:")
    p_pd = pd.Series(zp)
    s_pd = pd.Series(ks)
    cov_pd = p_pd.cov(s_pd)
    r_pd = p_pd.corr(s_pd, method='pearson')
    print(f"        pandas cov = {cov_pd:.4f}")
    print(f"        pandas r = {r_pd:.4f}")
    print()


def task_2():
    """
    Задача 2:
        Измерены значения IQ выборки студентов,
        обучающихся в местных технических вузах:
        131, 125, 115, 122, 131, 115, 107, 99, 125, 111.
        Известно, что в генеральной совокупности IQ распределен нормально.
        Найдите доверительный интервал для математического ожидания с надежностью 0.95.
    Решение:
        В задаче имеется выборка из 10 значений, среднее квадратичное генеральной совокупности неизвестно.
        Для поиска доверительного интервала используем распределение Стьюдента
        Размер выборки: 10
        Среднее выборки: 118.1
        Среднее квадратичное отклонение: 10.5457
        Табличное значение для доверительного интервала 95% при n = 10: 2.2622
        Доверительный интервал: Mсред +- Ta/2 * sigma/(n**0.5) = 118.1000 +- 7.5439
    Ответ:
        Доверительный интервал для математического ожидания с надежностью 0.95
        лежит в интервале [110.5561 ; 125.6439]
    """
    print("Задача 2")
    iq_list = [131, 125, 115, 122, 131, 115, 107, 99, 125, 111]
    mean = np.mean(iq_list)
    n = len(iq_list)
    sigma = np.std(iq_list, ddof=1)
    alpha = 0.95
    q = alpha + (1 - alpha) / 2
    t_a2 = t.ppf(q, n - 1)  # критерий Стьюдента для доверительного интервала 95% при n = 10

    delta = t_a2 * sigma / (n ** 0.5)
    iq_min = mean - delta
    iq_max = mean + delta
    print(f"    Размер выборки: {n}")
    print(f"    Среднее выборки: {mean}")
    print(f"    Среднее квадратичное отклонение: {sigma:.4f}")
    print(f"    Критерий Стьюдента для доверительного интервала 95% при n = {n}: {t_a2:.4f}")
    print(f"    Доверительный интервал: Mсред +- Ta/2 * sigma/(n**0.5) = {mean:.4f} +- {delta:.4f}")
    print(f"Ответ:")
    print(f"    Доверительный интервал для математического ожидания с надежностью 0.95")
    print(f"    лежит в интервале [{iq_min:.4f} ; {iq_max:.4f}]")
    print()


def task_3():
    """
    Задача 3:
        Известно, что рост футболистов в сборной распределен нормально
        с дисперсией генеральной совокупности, равной 25 кв.см. Объем выборки равен 27,
        среднее выборочное составляет 174.2. Найдите доверительный интервал для математического
        ожидания с надежностью 0.95.
    Решение:
        Т.к среднее квадратичное генеральной совокупности известно, можем использовать Z-распределение
        d = 25
        sigma = d ** 0.5 = 5
        n = 27
        Mсред = 174.2
        Для надежности 95%
        (1 - 0.95) / 2 = 0.05 / 2 = 0.025
        0.95 + 0.025 = 0.975
        Таблицная величина Z для 0.975 = 1.96

        Вычисляем интервал
        Mсред +- Z * sigma/(n**0.5)
        174.2 +- 1.96*5/(27**0.5) = 174.2 +- 1.8860
    Ответ:
        Доверительный интервал для оценки математического ожидания с надежностью 0.95:
        [172.3140 ; 176.0860]
    """
    print("Задача 3")
    d = 25
    sigma = d ** 0.5
    n = 27
    mean = 174.2
    alpha = 0.95
    q = alpha + (1 - alpha) / 2
    z_a2 = norm.ppf(q)
    print(f"    Mсред = {mean:.4f}")
    print(f"    n = {n}")
    print(f"    sigma = {sigma}")
    print(f"    q = {q:.4f}")
    print(f"    Za2 = {z_a2:.4f}")
    delta = z_a2 * sigma / (n ** 0.5)
    x_min = mean - delta
    x_max = mean + delta
    print(f"    Доверительный интервал: Mсред +- Za/2 * sigma/(n**0.5)")
    print(f"        или: {mean} +- {z_a2:.4f} * {sigma}/({n}**0.5) = {mean} +- {delta:.4f}")
    print(f"Ответ:")
    print(f"    доверительный интервал для математического ожидания с надежностью 0.95:")
    print(f"    [{x_min:.4f} ; {x_max:.4f}]")
    print()


task_1()
task_2()
task_3()
