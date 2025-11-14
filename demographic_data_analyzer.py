import pandas as pd

def calculate_demographic_data(print_data=True):

    df = pd.read_csv("adult.data.csv")

    race_count = df['race'].value_counts()

    average_age_men = round(df[df['sex'] == 'Male']['age'].mean(), 1)

    percentage_bachelors = round(
        (df['education'] == 'Bachelors').mean() * 100, 1
    )

    higher_edu = ['Bachelors', 'Masters', 'Doctorate']
    df_higher = df[df['education'].isin(higher_edu)]
    df_lower = df[~df['education'].isin(higher_edu)]

    higher_education_rich = round(
        (df_higher['salary'] == '>50K').mean() * 100, 1
    )

    lower_education_rich = round(
        (df_lower['salary'] == '>50K').mean() * 100, 1
    )

    min_work_hours = df['hours-per-week'].min()

    df_min_workers = df[df['hours-per-week'] == min_work_hours]
    rich_percentage = round(
        (df_min_workers['salary'] == '>50K').mean() * 100, 1
    )

    country_counts = df['native-country'].value_counts()
    rich_by_country = df[df['salary'] == '>50K']['native-country'].value_counts()

    highest_earning_country_percentage = round(
        (rich_by_country / country_counts * 100).fillna(0).max(), 1
    )

    highest_earning_country = (
        (rich_by_country / country_counts * 100)
        .fillna(0)
        .idxmax()
    )

    india_rich = df[
        (df['native-country'] == 'India') &
        (df['salary'] == '>50K')
    ]

    top_IN_occupation = india_rich['occupation'].value_counts().idxmax()

    if print_data:
        print("Número de personas por raza:\n", race_count)
        print("Edad promedio de los hombres:", average_age_men)
        print("Porcentaje con Bachelors:", percentage_bachelors)
        print("Educación superior (>50K):", higher_education_rich)
        print("Sin educación superior (>50K):", lower_education_rich)
        print("Horas mínimas por semana:", min_work_hours)
        print("Ricos entre quienes trabajan mínimo:", rich_percentage)
        print("País con mayor % de ricos:", highest_earning_country)
        print("Porcentaje:", highest_earning_country_percentage)
        print("Ocupación en India más común para ricos:", top_IN_occupation)

    return {
        'race_count': race_count,
        'average_age_men': average_age_men,
        'percentage_bachelors': percentage_bachelors,
        'higher_education_rich': higher_education_rich,
        'lower_education_rich': lower_education_rich,
        'min_work_hours': min_work_hours,
        'rich_percentage': rich_percentage,
        'highest_earning_country': highest_earning_country,
        'highest_earning_country_percentage': highest_earning_country_percentage,
        'top_IN_occupation': top_IN_occupation
    }
