# Association Rule Learning

**Association Rule Learning** helps uncover relationships between variables in large datasets. This technique is commonly used in market basket analysis to find product combinations that frequently co-occur in transactions, answering the classic question: "People who bought *this* also bought *that*." 

By the end of this section, you will understand how to implement and apply Association Rule Learning algorithms to discover patterns in transactional or categorical datasets, providing insights that can inform strategies such as cross-selling or promotional bundling.

## Overview of Association Rule Learning

We will cover the implementation and interpretation of the following Association Rule Learning models:

- **Apriori**: A foundational algorithm that efficiently mines frequent itemsets and identifies association rules by calculating support, confidence, and lift values.
- **Eclat**: A depth-first search algorithm that improves upon Apriori by using a vertical data format, making it more scalable for large datasets.

## Key Concepts
Association Rule Learning is based on identifying relationships between items in data:
- **Support**: The frequency with which an itemset appears in the dataset.
- **Confidence**: The likelihood that item Y is purchased when item X is purchased.
- **Lift**: Measures how much more likely item Y is purchased when item X is present, compared to random chance.

## Use Cases
- **Market Basket Analysis**: Discover patterns of product combinations in retail transactions (e.g., "Customers who buy milk often buy bread").
- **Recommendation Systems**: Improve recommendations in e-commerce platforms by identifying frequently co-occurring items.
- **Inventory Management**: Optimize stock levels by analyzing frequently bought itemsets.
  
This section will help you apply these models to real-world problems and improve decision-making based on patterns found in large datasets.
