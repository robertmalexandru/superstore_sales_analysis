# Superstore Sales Analysis

## Overview
Analysis of Superstore sales data to identify key business insights across sales trends, product performance, regional patterns, and profitability drivers.

## Setup and Data Loading
```python
# Essential imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure settings
plt.style.use("ggplot")
pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Load and prepare data
data = pd.read_csv("superstore.csv", encoding_errors="ignore")

# Convert dates and select relevant columns
data["order_date"] = pd.to_datetime(data["order_date"])
data["ship_date"] = pd.to_datetime(data["ship_date"])

essential_columns = [
    "row_id", "order_id", "order_date", "ship_date", "ship_mode", "customer_id", "customer_name", "segment",
    "country", "city", "state", "postal_code", "region", "product_id", "category", "sub-category", 
    "product_name", "sales", "quantity", "discount", "profit"
]
data = data[essential_columns]
```

## Feature Engineering
```python
# Create analytical features
data = data.assign(
    month=data["order_date"].dt.month,
    year=data["order_date"].dt.year,
    year_month=data["order_date"].dt.to_period("M"),
    discount_amount=data["sales"] * data["discount"],
    unit_price=data["sales"] / data["quantity"],
    gross_profit=data["sales"] * data["discount"] + data["profit"],
    fulfillment_time=data["ship_date"] - data["order_date"],
    profit_per_unit=data["profit"] / data["quantity"],
    profit_margin=(data["profit"] / data["sales"]) * 100,
    net_sales=data["sales"] - (data["discount"] * data["sales"])
)
```

## Sales Analysis

### Temporal Trends
```python
def plot_sales_trends(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Yearly trends
    yearly_data = data.groupby('year').agg({
        'sales': 'sum',
        'order_date': 'count'
    })
    
    yearly_data['sales'].plot(ax=ax1, color='#003f5c')
    ax1.set_title('Annual Sales')
    ax1.set_ylabel('Total Sales ($)')
    
    # Monthly trends
    monthly_data = data.groupby('year_month')['sales'].sum()
    monthly_data.plot(ax=ax2, color='#003f5c')
    ax2.set_title('Monthly Sales Trend')
    ax2.set_ylabel('Sales ($)')
    
    plt.tight_layout()
    return fig

sales_trends = plot_sales_trends(data)
plt.show()
```

### Product Performance
```python
def analyze_product_performance(data):
    # Calculate product metrics
    product_metrics = data.groupby(['category', 'sub-category']).agg({
        'sales': 'sum',
        'profit': 'sum',
        'quantity': 'sum',
        'profit_margin': 'mean'
    }).round(2)
    
    # Plot performance
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data,
        x='sales',
        y='sub-category',
        hue='category',
        palette=['#003049', '#d62828', '#f77f00']
    )
    plt.title('Sales by Product Category')
    plt.tight_layout()
    
    return product_metrics

product_performance = analyze_product_performance(data)
print("\nProduct Performance Metrics:")
print(product_performance)
```

### Regional Analysis
```python
def analyze_regional_performance(data):
    # Regional sales over time
    regional_sales = data.pivot_table(
        index='year_month',
        columns='region',
        values='sales',
        aggfunc='sum'
    )
    
    # Plot regional trends
    plt.figure(figsize=(12, 6))
    regional_sales.plot(linewidth=1.5)
    plt.title('Regional Sales Trends')
    plt.xlabel('Date')
    plt.ylabel('Sales ($)')
    plt.legend(title='Region')
    plt.tight_layout()
    
    return regional_sales

regional_analysis = analyze_regional_performance(data)
```

### Profitability Analysis
```python
def analyze_profitability(data):
    # Calculate profitability metrics
    profit_metrics = data.groupby(['category', 'sub-category']).agg({
        'profit': ['sum', 'mean'],
        'profit_margin': 'mean',
        'discount_amount': 'sum'
    }).round(2)
    
    # Plot profit margins
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x='category',
        y='profit_margin',
        palette=['#003049', '#d62828', '#f77f00']
    )
    plt.title('Profit Margin Distribution by Category')
    plt.tight_layout()
    
    return profit_metrics

profitability = analyze_profitability(data)
print("\nProfitability Metrics:")
print(profitability)
```

## Key Insights

1. Sales Patterns
   - Consistent yearly growth with seasonal peaks in Nov-Dec
   - Strong back-to-school sales in September
   - Higher variability in March and Q4

2. Product Performance
   - Technology: Phones lead sales, copiers underperform
   - Furniture: Chairs dominate, bookcases struggle
   - Office Supplies: Storage and binders show steady growth

3. Regional Trends
   - West and East regions lead in sales
   - South shows opportunity for growth
   - Central region maintains steady performance

4. Profitability
   - Overall profit margin >10%
   - High-margin categories: Copiers, Labels, Furnishings
   - Discount impact significant on Tables and Bookcases

## Recommendations

1. Inventory Management
   - Optimize stock levels for seasonal peaks
   - Implement demand forecasting for high-variability periods

2. Regional Strategy
   - Develop targeted growth plans for South region
   - Leverage successful practices from West/East

3. Product Portfolio
   - Review underperforming sub-categories
   - Optimize discount strategy for high-margin products

4. Operational Efficiency
   - Focus on reducing fulfillment times
   - Analyze and optimize shipping costs
