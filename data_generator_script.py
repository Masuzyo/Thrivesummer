#!/usr/bin/env python3
"""
Dataset Generator Script for Statistics Course
Generates realistic datasets for various statistical analysis exercises
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import argparse
import os

class DatasetGenerator:
    """Class to generate various types of datasets for statistical analysis"""
    
    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility"""
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
    
    def generate_student_dataset(self, n=1000, save_csv=True, filename="students_extended.csv"):
        """
        Generate extended student dataset with more realistic relationships
        
        Parameters:
        n (int): Number of students to generate
        save_csv (bool): Whether to save to CSV file
        filename (str): Name of CSV file to save
        
        Returns:
        pd.DataFrame: Generated student dataset
        """
        print(f"Generating student dataset with {n} entries...")
        
        # Generate correlated data for realism
        # Age influences other variables
        ages = np.random.normal(20, 2.5, n)
        ages = np.clip(ages, 17, 28).astype(int)
        
        # Majors with different proportions
        majors = np.random.choice(['Computer Science', 'Mathematics', 'Physics', 'Biology', 
                                  'Chemistry', 'Engineering', 'Psychology', 'Economics'], 
                                 n, p=[0.20, 0.12, 0.08, 0.15, 0.10, 0.18, 0.12, 0.05])
        
        # GPA influenced by study hours and major difficulty
        base_gpa = np.random.normal(3.0, 0.6, n)
        
        # Major difficulty adjustments
        major_difficulty = {
            'Computer Science': 0.1, 'Mathematics': 0.2, 'Physics': 0.15, 'Engineering': 0.1,
            'Chemistry': 0.05, 'Biology': 0.0, 'Psychology': -0.1, 'Economics': -0.05
        }
        
        gpa_adjustments = [major_difficulty[major] for major in majors]
        gpas = base_gpa + np.array(gpa_adjustments) + np.random.normal(0, 0.2, n)
        gpas = np.clip(gpas, 0.0, 4.0)
        
        # Study hours influenced by GPA and major
        base_study_hours = np.random.poisson(12, n)
        gpa_influence = (gpas - 2.5) * 5  # Higher GPA = more study hours
        study_hours = base_study_hours + gpa_influence + np.random.normal(0, 2, n)
        study_hours = np.clip(study_hours, 2, 40).astype(int)
        
        # Add more variables
        # Gender
        genders = np.random.choice(['Male', 'Female', 'Other'], n, p=[0.48, 0.50, 0.02])
        
        # Year in school
        years = np.random.choice(['Freshman', 'Sophomore', 'Junior', 'Senior'], 
                               n, p=[0.28, 0.26, 0.24, 0.22])
        
        # Employment status (influenced by year)
        employment_prob = {'Freshman': [0.7, 0.25, 0.05], 'Sophomore': [0.6, 0.35, 0.05],
                          'Junior': [0.4, 0.5, 0.1], 'Senior': [0.3, 0.5, 0.2]}
        
        employment = []
        for year in years:
            employment.append(np.random.choice(['Unemployed', 'Part-time', 'Full-time'], 
                                             p=employment_prob[year]))
        
        # Income (influenced by employment and year)
        incomes = []
        for emp, year in zip(employment, years):
            if emp == 'Unemployed':
                income = 0
            elif emp == 'Part-time':
                base = 8000 if year in ['Freshman', 'Sophomore'] else 12000
                income = np.random.normal(base, 2000)
            else:  # Full-time
                base = 25000 if year in ['Freshman', 'Sophomore'] else 35000
                income = np.random.normal(base, 5000)
            incomes.append(max(0, income))
        
        # Create dataset
        student_data = {
            'StudentID': range(1, n + 1),
            'Age': ages,
            'Gender': genders,
            'Major': majors,
            'Year': years,
            'GPA': np.round(gpas, 2),
            'Study_Hours_Per_Week': study_hours,
            'Employment_Status': employment,
            'Annual_Income': np.round(incomes, 2),
            'Credits_Enrolled': np.random.choice([12, 15, 18, 21], n, p=[0.1, 0.4, 0.4, 0.1]),
            'Has_Scholarship': np.random.choice([True, False], n, p=[0.3, 0.7]),
            'Lives_On_Campus': np.random.choice([True, False], n, p=[0.6, 0.4])
        }
        
        df = pd.DataFrame(student_data)
        
        if save_csv:
            df.to_csv(filename, index=False)
            print(f"Student dataset saved to {filename}")
        
        return df
    
    def generate_survey_dataset(self, n=500, save_csv=True, filename="survey_extended.csv"):
        """
        Generate extended survey dataset with realistic response patterns
        
        Parameters:
        n (int): Number of survey responses
        save_csv (bool): Whether to save to CSV
        filename (str): Filename for CSV
        
        Returns:
        pd.DataFrame: Generated survey dataset
        """
        print(f"Generating survey dataset with {n} responses...")
        
        # Demographics
        age_groups = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65', '65+'], 
                                    n, p=[0.25, 0.25, 0.20, 0.15, 0.10, 0.05])
        
        education_levels = np.random.choice(['High School', 'Some College', 'Bachelor', 'Master', 'PhD'], 
                                          n, p=[0.25, 0.20, 0.35, 0.15, 0.05])
        
        income_levels = np.random.choice(['<25k', '25-50k', '50-75k', '75-100k', '100k+'], 
                                       n, p=[0.20, 0.25, 0.25, 0.20, 0.10])
        
        # Location
        regions = np.random.choice(['Northeast', 'Southeast', 'Midwest', 'Southwest', 'West'], 
                                 n, p=[0.18, 0.22, 0.24, 0.16, 0.20])
        
        # Product/Service ratings (1-10 scale)
        # Create realistic correlation between different rating aspects
        base_satisfaction = np.random.normal(7, 2, n)
        base_satisfaction = np.clip(base_satisfaction, 1, 10)
        
        # Correlated ratings
        quality_rating = base_satisfaction + np.random.normal(0, 0.5, n)
        quality_rating = np.clip(quality_rating, 1, 10)
        
        price_rating = base_satisfaction + np.random.normal(-0.5, 1, n)  # Price slightly lower
        price_rating = np.clip(price_rating, 1, 10)
        
        service_rating = base_satisfaction + np.random.normal(0.2, 0.8, n)
        service_rating = np.clip(service_rating, 1, 10)
        
        # Likelihood to recommend (influenced by overall satisfaction)
        recommend_prob = (base_satisfaction - 1) / 9  # Convert to 0-1 probability
        would_recommend = np.random.binomial(1, recommend_prob, n)
        
        # Usage frequency
        usage_frequency = np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely', 'Never'], 
                                         n, p=[0.15, 0.25, 0.30, 0.25, 0.05])
        
        # Time as customer (in months)
        time_as_customer = np.random.exponential(18, n)
        time_as_customer = np.clip(time_as_customer, 1, 120).astype(int)
        
        survey_data = {
            'ResponseID': range(1, n + 1),
            'Age_Group': age_groups,
            'Education_Level': education_levels,
            'Income_Level': income_levels,
            'Region': regions,
            'Overall_Satisfaction': np.round(base_satisfaction, 1),
            'Quality_Rating': np.round(quality_rating, 1),
            'Price_Rating': np.round(price_rating, 1),
            'Service_Rating': np.round(service_rating, 1),
            'Would_Recommend': would_recommend,
            'Usage_Frequency': usage_frequency,
            'Months_As_Customer': time_as_customer,
            'Number_Of_Purchases': np.random.poisson(8, n),
            'Preferred_Contact': np.random.choice(['Email', 'Phone', 'Text', 'Mail'], 
                                                n, p=[0.45, 0.25, 0.25, 0.05])
        }
        
        df = pd.DataFrame(survey_data)
        
        if save_csv:
            df.to_csv(filename, index=False)
            print(f"Survey dataset saved to {filename}")
        
        return df
    
    def generate_daily_habits_dataset(self, n_days=365, n_people=50, save_csv=True, 
                                    filename="daily_habits_extended.csv"):
        """
        Generate extended daily habits dataset with multiple people over time
        
        Parameters:
        n_days (int): Number of days to generate data for
        n_people (int): Number of people to track
        save_csv (bool): Whether to save to CSV
        filename (str): Filename for CSV
        
        Returns:
        pd.DataFrame: Generated habits dataset
        """
        print(f"Generating daily habits dataset for {n_people} people over {n_days} days...")
        
        all_data = []
        
        for person_id in range(1, n_people + 1):
            # Each person has baseline habits with some variation
            
            # Personal baseline characteristics
            baseline_sleep = np.random.normal(7.5, 1, 1)[0]
            baseline_exercise = np.random.exponential(25, 1)[0]
            baseline_mood = np.random.normal(7, 1, 1)[0]
            
            # Generate daily data for this person
            for day in range(1, n_days + 1):
                # Add seasonal and weekly patterns
                seasonal_factor = np.sin(2 * np.pi * day / 365) * 0.5  # Yearly cycle
                weekly_factor = np.sin(2 * np.pi * day / 7) * 0.3      # Weekly cycle
                
                # Sleep hours with personal baseline and patterns
                sleep_hours = (baseline_sleep + seasonal_factor + 
                             np.random.normal(0, 0.8) + 
                             (1 if day % 7 in [6, 0] else 0))  # Weekend effect
                sleep_hours = np.clip(sleep_hours, 4, 12)
                
                # Coffee consumption (inversely related to sleep quality)
                sleep_quality = sleep_hours / 12  # Normalized sleep quality
                coffee_lambda = 3 - sleep_quality + np.random.normal(0, 0.5)
                cups_coffee = np.random.poisson(max(0, coffee_lambda))
                
                # Exercise minutes
                exercise_minutes = (baseline_exercise + weekly_factor * 10 + 
                                  np.random.normal(0, 15))
                exercise_minutes = max(0, int(exercise_minutes))
                
                # Screen time (inversely related to exercise)
                base_screen_time = 6 + (40 - exercise_minutes) / 10
                screen_time = base_screen_time + np.random.normal(0, 1.5)
                screen_time = np.clip(screen_time, 2, 14)
                
                # Mood rating (influenced by sleep, exercise, and screen time)
                mood_factors = (sleep_hours - 7) * 0.3 + (exercise_minutes - 30) * 0.02 - (screen_time - 6) * 0.1
                mood_rating = baseline_mood + mood_factors + np.random.normal(0, 1)
                mood_rating = np.clip(mood_rating, 1, 10)
                
                # Water intake
                water_glasses = np.random.poisson(8) + (1 if exercise_minutes > 30 else 0)
                
                # Steps (related to exercise)
                base_steps = 6000 + exercise_minutes * 50
                steps = int(base_steps + np.random.normal(0, 2000))
                steps = max(1000, steps)
                
                # Day of week
                day_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                              'Friday', 'Saturday', 'Sunday'][day % 7]
                
                daily_data = {
                    'PersonID': person_id,
                    'Day': day,
                    'DayOfWeek': day_of_week,
                    'Hours_Sleep': round(sleep_hours, 1),
                    'Cups_Coffee': cups_coffee,
                    'Exercise_Minutes': exercise_minutes,
                    'Screen_Time_Hours': round(screen_time, 1),
                    'Mood_Rating': round(mood_rating, 1),
                    'Water_Glasses': water_glasses,
                    'Steps': steps,
                    'Calories_Consumed': int(np.random.normal(2000, 300)),
                    'Productivity_Rating': round(np.clip(np.random.normal(7, 2), 1, 10), 1)
                }
                
                all_data.append(daily_data)
        
        df = pd.DataFrame(all_data)
        
        if save_csv:
            df.to_csv(filename, index=False)
            print(f"Daily habits dataset saved to {filename}")
        
        return df
    
    def generate_sales_dataset(self, n=2000, save_csv=True, filename="sales_data.csv"):
        """
        Generate sales dataset for business analytics
        
        Parameters:
        n (int): Number of sales records
        save_csv (bool): Whether to save to CSV
        filename (str): Filename for CSV
        
        Returns:
        pd.DataFrame: Generated sales dataset
        """
        print(f"Generating sales dataset with {n} records...")
        
        # Products
        products = ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Monitor', 
                   'Keyboard', 'Mouse', 'Webcam', 'Speaker', 'Charger']
        
        # Product categories and base prices
        product_info = {
            'Laptop': {'category': 'Electronics', 'base_price': 800, 'std': 200},
            'Smartphone': {'category': 'Electronics', 'base_price': 600, 'std': 150},
            'Tablet': {'category': 'Electronics', 'base_price': 400, 'std': 100},
            'Headphones': {'category': 'Accessories', 'base_price': 100, 'std': 50},
            'Monitor': {'category': 'Electronics', 'base_price': 300, 'std': 100},
            'Keyboard': {'category': 'Accessories', 'base_price': 50, 'std': 20},
            'Mouse': {'category': 'Accessories', 'base_price': 30, 'std': 15},
            'Webcam': {'category': 'Electronics', 'base_price': 80, 'std': 30},
            'Speaker': {'category': 'Electronics', 'base_price': 120, 'std': 60},
            'Charger': {'category': 'Accessories', 'base_price': 25, 'std': 10}
        }
        
        # Generate sales data
        sales_data = []
        
        for i in range(1, n + 1):
            # Random date in the last 2 years
            start_date = datetime.now() - timedelta(days=730)
            random_days = np.random.randint(0, 730)
            sale_date = start_date + timedelta(days=random_days)
            
            # Select product
            product = np.random.choice(products)
            product_details = product_info[product]
            
            # Price with some variation
            price = np.random.normal(product_details['base_price'], product_details['std'])
            price = max(10, round(price, 2))  # Minimum price of $10
            
            # Quantity (most sales are single items, some bulk)
            quantity = np.random.choice([1, 2, 3, 4, 5], p=[0.6, 0.2, 0.1, 0.05, 0.05])
            
            # Customer info
            customer_age = np.random.randint(18, 80)
            customer_gender = np.random.choice(['Male', 'Female', 'Other'], p=[0.48, 0.50, 0.02])
            
            # Sales channel
            channel = np.random.choice(['Online', 'In-Store', 'Phone'], p=[0.6, 0.35, 0.05])
            
            # Region
            region = np.random.choice(['North', 'South', 'East', 'West'], p=[0.25, 0.25, 0.25, 0.25])
            
            # Salesperson (only for in-store and phone)
            if channel in ['In-Store', 'Phone']:
                salesperson = f"Sales_{np.random.randint(1, 21)}"  # 20 salespeople
            else:
                salesperson = 'Online'
            
            # Discount (seasonal patterns)
            month = sale_date.month
            if month in [11, 12]:  # Holiday season
                discount_prob = 0.4
            elif month in [6, 7, 8]:  # Summer sale
                discount_prob = 0.3
            else:
                discount_prob = 0.1
            
            has_discount = np.random.choice([True, False], p=[discount_prob, 1-discount_prob])
            discount_percent = np.random.uniform(5, 25) if has_discount else 0
            
            # Final price after discount
            final_price = price * (1 - discount_percent/100)
            total_amount = final_price * quantity
            
            sale_record = {
                'SaleID': i,
                'Date': sale_date.strftime('%Y-%m-%d'),
                'Product': product,
                'Category': product_details['category'],
                'Price': round(price, 2),
                'Quantity': quantity,
                'Discount_Percent': round(discount_percent, 1),
                'Final_Price': round(final_price, 2),
                'Total_Amount': round(total_amount, 2),
                'Customer_Age': customer_age,
                'Customer_Gender': customer_gender,
                'Sales_Channel': channel,
                'Region': region,
                'Salesperson': salesperson,
                'Month': month,
                'Quarter': (month - 1) // 3 + 1,
                'Weekday': sale_date.strftime('%A')
            }
            
            sales_data.append(sale_record)
        
        df = pd.DataFrame(sales_data)
        
        if save_csv:
            df.to_csv(filename, index=False)
            print(f"Sales dataset saved to {filename}")
        
        return df
    
    def generate_all_datasets(self, student_n=1000, survey_n=500, habits_days=365, 
                            habits_people=50, sales_n=2000):
        """
        Generate all datasets at once
        
        Parameters:
        student_n (int): Number of student records
        survey_n (int): Number of survey responses
        habits_days (int): Number of days for habits data
        habits_people (int): Number of people for habits data
        sales_n (int): Number of sales records
        
        Returns:
        dict: Dictionary containing all generated datasets
        """
        print("=" * 50)
        print("GENERATING ALL DATASETS")
        print("=" * 50)
        
        datasets = {}
        
        datasets['students'] = self.generate_student_dataset(student_n)
        datasets['survey'] = self.generate_survey_dataset(survey_n)
        datasets['habits'] = self.generate_daily_habits_dataset(habits_days, habits_people)
        datasets['sales'] = self.generate_sales_dataset(sales_n)
        
        print("\n" + "=" * 50)
        print("ALL DATASETS GENERATED SUCCESSFULLY!")
        print("=" * 50)
        
        # Print summary
        for name, df in datasets.items():
            print(f"{name.capitalize()} dataset: {len(df)} records, {len(df.columns)} columns")
        
        return datasets

def main():
    """Main function to run the dataset generator"""
    parser = argparse.ArgumentParser(description='Generate datasets for statistics course')
    parser.add_argument('--dataset', choices=['students', 'survey', 'habits', 'sales', 'all'], 
                       default='all', help='Which dataset to generate')
    parser.add_argument('--size', type=int, default=1000, 
                       help='Size of dataset (interpretation depends on dataset type)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    parser.add_argument('--no-save', action='store_true', 
                       help='Don\'t save to CSV files')
    
    args = parser.parse_args()
    
    # Create generator
    generator = DatasetGenerator(seed=args.seed)
    
    # Generate requested dataset(s)
    if args.dataset == 'students':
        generator.generate_student_dataset(args.size, save_csv=not args.no_save)
    elif args.dataset == 'survey':
        generator.generate_survey_dataset(args.size, save_csv=not args.no_save)
    elif args.dataset == 'habits':
        generator.generate_daily_habits_dataset(args.size, save_csv=not args.no_save)
    elif args.dataset == 'sales':
        generator.generate_sales_dataset(args.size, save_csv=not args.no_save)
    elif args.dataset == 'all':
        generator.generate_all_datasets()

if __name__ == "__main__":
    # If running as script, use command line arguments
    if len(os.sys.argv) > 1:
        main()
    else:
        # If running interactively, generate all datasets with default parameters
        print("Running in interactive mode - generating all datasets...")
        generator = DatasetGenerator()
        datasets = generator.generate_all_datasets()
        
        # Display sample data
        print("\n" + "=" * 50)
        print("SAMPLE DATA PREVIEW")
        print("=" * 50)
        
        for name, df in datasets.items():
            print(f"\n{name.upper()} DATASET (first 5 rows):")
            print(df.head())
            print(f"Shape: {df.shape}")
