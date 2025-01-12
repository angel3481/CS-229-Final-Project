import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv('car_prices_4.csv')

# Set the style for better visualizations
plt.style.use('seaborn-v0_8')

def plot_distributions():
    # Create a figure with multiple subplots - made even taller
    fig = plt.figure(figsize=(20, 45))  # Store figure in variable
    
    # Set even smaller font sizes globally
    plt.rcParams.update({
        'font.size': 6,          
        'axes.labelsize': 6,     
        'axes.titlesize': 8,     
        'xtick.labelsize': 5,    
        'ytick.labelsize': 5     
    })
    
    # Create GridSpec to have more control over spacing
    gs = fig.add_gridspec(5, 2, hspace=0.4, wspace=0.3)  # Explicit spacing between plots
    
    # Modify subplot calls to use GridSpec
    # Year distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.histplot(data=df, x='year', bins=15, ax=ax1)
    ax1.set_title('Distribution of Car Years')
    
    # Price distribution
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=df, x='price', bins=50, ax=ax2)
    ax2.set_title('Distribution of Prices')
    
    # Odometer distribution
    ax3 = fig.add_subplot(gs[1, 0])
    sns.histplot(data=df, x='odometer', bins=40, ax=ax3)
    ax3.set_title('Distribution of Odometer Readings')
    
    # Condition distribution
    ax4 = fig.add_subplot(gs[1, 1])
    sns.histplot(data=df, x='condition', bins=20, ax=ax4)
    ax4.set_title('Distribution of Condition Ratings')
    
    # Make distribution
    ax5 = fig.add_subplot(gs[2, 0])
    sns.countplot(data=df, y='make', order=df['make'].value_counts().index[:20], ax=ax5)
    ax5.set_title('Top 20 Car Makes')
    
    # Body distribution
    ax6 = fig.add_subplot(gs[2, 1])
    sns.countplot(data=df, y='body', ax=ax6)
    ax6.set_title('Distribution of Body Types')
    
    # Transmission distribution
    ax7 = fig.add_subplot(gs[3, 0])
    sns.countplot(data=df, x='transmission', ax=ax7)
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    ax7.set_title('Distribution of Transmission Types')
    
    # State distribution
    ax8 = fig.add_subplot(gs[3, 1])
    sns.countplot(data=df, y='state', order=df['state'].value_counts().index[:20], ax=ax8)
    ax8.set_title('Top 20 States')
    
    # Exterior color distribution
    ax9 = fig.add_subplot(gs[4, 0])
    sns.countplot(data=df, y='exterior_color', order=df['exterior_color'].value_counts().index[:15], ax=ax9)
    ax9.set_title('Distribution of Exterior Colors')
    
    # Interior color distribution
    ax10 = fig.add_subplot(gs[4, 1])
    sns.countplot(data=df, y='interior_color', order=df['interior_color'].value_counts().index[:15], ax=ax10)
    ax10.set_title('Distribution of Interior Colors')
    
    # Adjust the layout
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.8, wspace=0.3)
    plt.show()

def analyze_price_distribution():
    # Create figure with higher resolution
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
    
    # Regular histogram with many more bins (250 instead of 100)
    sns.histplot(data=df, x='price', bins=250, ax=ax1)
    ax1.set_title('Detailed Distribution of Car Prices')
    ax1.set_xlabel('Price ($)')
    ax1.set_ylabel('Count')
    
    # Format x-axis with thousand separators
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Add grid for better readability
    ax1.grid(True, alpha=0.3)
    
    # Zoom in on the main distribution (excluding extreme outliers)
    price_99th = df['price'].quantile(0.99)
    ax1.set_xlim(0, price_99th)
    
    # Log-scale histogram with same number of bins
    sns.histplot(data=df, x='price', bins=250, ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Price Distribution (Log Scale)')
    ax2.set_xlabel('Price ($)')
    ax2.set_ylabel('Count (Log Scale)')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nDetailed Price Statistics:")
    print(f"Mean Price: ${df['price'].mean():,.2f}")
    print(f"Median Price: ${df['price'].median():,.2f}")
    print(f"Standard Deviation: ${df['price'].std():,.2f}")
    print(f"Skewness: {df['price'].skew():,.2f}")
    print(f"Kurtosis: {df['price'].kurtosis():,.2f}")
    print(f"\nFine-grained Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: ${df['price'].quantile(p/100):,.2f}")

if __name__ == "__main__":
    # plot_distributions()  # Comment out or keep based on your needs
    analyze_price_distribution()
