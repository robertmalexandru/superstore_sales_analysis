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
data["Order Date"] = pd.to_datetime(data["Order Date"])
data["Ship Date"] = pd.to_datetime(data["Ship Date"])

essential_columns = [
    "Order ID", "Order Date", "Ship Date", "Ship Mode", "Segment",
    "City", "State", "Region", "Category", "Sub-Category", 
    "Product Name", "Sales", "Quantity", "Discount", "Profit"
]
data = data[essential_columns]
```

## Feature Engineering
```python
# Create analytical features
data = data.assign(
    month=data["Order Date"].dt.month,
    year=data["Order Date"].dt.year,
    year_month=data["Order Date"].dt.to_period("M"),
    discount_amount=data["Sales"] * data["Discount"],
    unit_price=data["Sales"] / data["Quantity"],
    gross_profit=data["Sales"] * data["Discount"] + data["Profit"],
    fulfillment_time=data["Ship Date"] - data["Order Date"],
    profit_per_unit=data["Profit"] / data["Quantity"],
    profit_margin=(data["Profit"] / data["Sales"]) * 100,
    net_sales=data["Sales"] - (data["Discount"] * data["Sales"])
)
```

## Sales Analysis

### Temporal Trends
```python
def plot_sales_trends(data):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Yearly trends
    yearly_data = data.groupby('year').agg({
        'Sales': 'sum',
        'Order Date': 'count'
    })
    
    yearly_data['Sales'].plot(ax=ax1, color='#003f5c')
    ax1.set_title('Annual Sales')
    ax1.set_ylabel('Total Sales ($)')
    
    # Monthly trends
    monthly_data = data.groupby('year_month')['Sales'].sum()
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
    product_metrics = data.groupby(['Category', 'Sub-Category']).agg({
        'Sales': 'sum',
        'Profit': 'sum',
        'Quantity': 'sum',
        'profit_margin': 'mean'
    }).round(2)
    
    # Plot performance
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=data,
        x='Sales',
        y='Sub-Category',
        hue='Category',
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
        columns='Region',
        values='Sales',
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
    profit_metrics = data.groupby(['Category', 'Sub-Category']).agg({
        'Profit': ['sum', 'mean'],
        'profit_margin': 'mean',
        'discount_amount': 'sum'
    }).round(2)
    
    # Plot profit margins
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=data,
        x='Category',
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
