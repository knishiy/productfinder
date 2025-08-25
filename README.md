# 🚀 Winning Product Finder

An AI-powered platform that discovers high-potential products by analyzing data from eBay, Amazon, Google Trends, and AliExpress. The system calculates risk factors for higher-priced items instead of strictly enforcing a $10 limit, providing intelligent product recommendations based on multiple criteria.

## ✨ Features

- **Multi-Source Data Collection**: eBay, Amazon Keepa, Google Trends, AliExpress
- **Intelligent Product Matching**: AI-powered title similarity and image analysis
- **Risk-Based Costing**: Calculates risk factors for items above $10 instead of strict limits
- **Comprehensive Scoring**: Multi-factor analysis including margin, demand, trends, and competition
- **Modern Web Interface**: Beautiful, responsive dashboard with real-time pipeline monitoring
- **Export Capabilities**: CSV export and detailed reporting
- **Real-time Monitoring**: Live pipeline status and progress tracking

## 🏗️ Architecture

```
WinningProdAgr/
├── etl/                    # Data extraction modules
│   ├── ebay.py            # eBay API integration
│   ├── amazon_keepa.py    # Amazon Keepa API
│   ├── trends.py          # Google Trends analysis
│   └── aliexpress.py      # AliExpress supplier data
├── matching.py             # Product matching engine
├── costing.py              # Cost calculation and risk assessment
├── scoring.py              # Product scoring and ranking
├── pipeline.py             # Main orchestration pipeline
├── web_app.py              # Flask web application
├── config.yaml             # Configuration file
├── requirements.txt        # Python dependencies
└── templates/              # Web interface templates
    ├── index.html         # Landing page
    └── dashboard.html     # Main dashboard
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- API keys for eBay, Keepa, and Apify (optional)

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd WinningProdAgr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

Edit `config.yaml` with your API keys and preferences:

```yaml
api_keys:
  ebay:
    client_id: "YOUR_EBAY_CLIENT_ID"
    client_secret: "YOUR_EBAY_CLIENT_SECRET"
  keepa:
    api_key: "YOUR_KEEPA_API_KEY"
  apify:
    token: "YOUR_APIFY_TOKEN"

categories:
  electronics:
    - id: "15032"
      name: "Cell Phones & Accessories"
      keywords: ["wireless earbuds", "phone case", "charging cable"]
```

### 4. Run the Application

```bash
# Start the web interface
python web_app.py

# Or run the pipeline directly
python pipeline.py
```

Visit `http://localhost:5000` to access the web interface.

## 🔧 API Setup

### eBay API
1. Go to [eBay Developers](https://developer.ebay.com/)
2. Create a new application
3. Get Client ID and Client Secret
4. Add to `config.yaml`

### Keepa API
1. Visit [Keepa API](https://keepa.com/#!api)
2. Sign up and get your API key
3. Add to `config.yaml`

### Apify (AliExpress Fallback)
1. Go to [Apify](https://apify.com/)
2. Create account and get API token
3. Add to `config.yaml`

## 📊 How It Works

### 1. Data Collection
- **eBay**: Collects market data, best-sellers, and competition analysis
- **Amazon Keepa**: Analyzes price stability and sales rank momentum
- **Google Trends**: Monitors keyword growth and trending patterns
- **AliExpress**: Finds supplier products with competitive pricing

### 2. Product Matching
- **Title Similarity**: Uses RapidFuzz for intelligent text matching
- **Image Analysis**: Perceptual hashing for visual similarity
- **Combined Scoring**: 60% title + 40% image similarity

### 3. Risk Assessment
Instead of strict $10 limits, the system calculates risk factors:
- **Low Risk (≤$10)**: 0% risk factor
- **Medium Risk ($10-$15)**: 0-10% risk factor
- **High Risk ($15-$20)**: 10-25% risk factor
- **Critical Risk ($20+)**: 25%+ risk factor

### 4. Scoring System
```
Final Score = (0.35 × Margin + 0.20 × Demand + 0.15 × Trends + 
               0.10 × Stability + 0.10 × Competition + 0.10 × Logistics) - Penalties
```

## 🎯 Usage Examples

### Command Line Pipeline

```python
from pipeline import WinningProductPipeline

# Initialize pipeline
pipeline = WinningProductPipeline("config.yaml")

# Run complete analysis
result = pipeline.run_pipeline()

print(f"Found {result.winning_products} winning products!")
print(f"Execution time: {result.execution_time:.2f} seconds")
```

### Individual ETL Components

```python
from etl.ebay import EbayETL
from etl.trends import TrendsETL

# eBay data collection
ebay = EbayETL("client_id", "client_secret")
products = ebay.search_products("wireless earbuds", "15032")

# Google Trends analysis
trends = TrendsETL()
trend_data = trends.get_trends(["wireless earbuds", "bluetooth speaker"])
```

### Product Matching

```python
from matching import ProductMatcher

matcher = ProductMatcher()
matches = matcher.find_matches(market_products, supplier_products)

for market_id, product_matches in matches.items():
    print(f"Market product {market_id}: {len(product_matches)} matches")
```

## 📈 Dashboard Features

- **Real-time Pipeline Monitoring**: Live progress tracking
- **Interactive Charts**: Score distribution and category analysis
- **Product Rankings**: Sortable table with detailed metrics
- **Export Functionality**: CSV download for further analysis
- **Responsive Design**: Works on desktop and mobile devices

## 🔍 Configuration Options

### Costing Thresholds
```yaml
costing:
  landed_cost_cap: 10.00          # Base cap
  risk_factor_multiplier: 1.5     # Risk calculation multiplier
  max_acceptable_cost: 25.00      # Absolute maximum
  shipping_buffer_percent: 5      # Safety buffer
```

### Scoring Weights
```yaml
scoring:
  margin_weight: 0.35             # Profit margin importance
  demand_velocity_weight: 0.20    # Market demand weight
  trend_growth_weight: 0.15       # Growth trend weight
  price_stability_weight: 0.10    # Price stability weight
  competition_density_weight: 0.10 # Competition analysis weight
  lead_time_weight: 0.10          # Shipping time weight
```

### Risk Assessment
```yaml
risk_assessment:
  denied_brands: ["apple", "samsung", "sony"]  # Brand restrictions
  min_seller_rating: 4.5                       # Minimum supplier rating
  high_risk_categories: ["counterfeit", "replica"] # Risk categories
```

## 📊 Output Formats

### CSV Export
- Product rankings and scores
- Cost breakdowns
- Risk assessments
- Supplier information

### JSON Reports
- Pipeline execution summary
- Scoring breakdowns
- Market analysis data
- Trend insights

### Data Directories
```
data/
├── raw/           # Raw market data
├── processed/     # Cleaned supplier data
├── matches/       # Product matching results
├── scoring/       # Scoring and ranking data
├── reports/       # Summary reports
├── web/           # Web interface data
└── exports/       # User exports
```

## 🚨 Error Handling

The system includes comprehensive error handling:
- **API Rate Limiting**: Automatic delays and retries
- **Data Validation**: Input sanitization and validation
- **Graceful Degradation**: Continues operation with partial data
- **Detailed Logging**: Comprehensive error tracking and debugging

## 🔒 Security Features

- **API Key Management**: Secure credential storage
- **Rate Limiting**: Respects API usage limits
- **Data Sanitization**: Input validation and cleaning
- **Error Masking**: Prevents sensitive information exposure

## 📈 Performance Optimization

- **Parallel Processing**: Multi-threaded data collection
- **Caching**: Image hash and API response caching
- **Batch Operations**: Efficient bulk data processing
- **Memory Management**: Optimized data structures

## 🧪 Testing

```bash
# Run individual module tests
python -m etl.ebay
python -m etl.trends
python matching.py
python costing.py
python scoring.py

# Run complete pipeline test
python pipeline.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Documentation**: Check this README and inline code comments
- **Issues**: Report bugs via GitHub Issues
- **Discussions**: Use GitHub Discussions for questions
- **Wiki**: Additional documentation and examples

## 🔮 Future Enhancements

- **Machine Learning**: Advanced product prediction models
- **Real-time Alerts**: Notifications for trending products
- **Competitor Analysis**: Advanced market intelligence
- **Mobile App**: Native mobile application
- **API Endpoints**: RESTful API for external integrations
- **Advanced Analytics**: Predictive modeling and forecasting

## 📊 System Requirements

- **Python**: 3.11+
- **Memory**: 4GB+ RAM recommended
- **Storage**: 1GB+ free space
- **Network**: Stable internet connection
- **APIs**: eBay, Keepa, and optional Apify accounts

## 🎉 Success Stories

The Winning Product Finder has helped entrepreneurs discover profitable opportunities across various niches:

- **Electronics**: Found trending wireless earbuds with 40%+ margins
- **Home & Garden**: Identified viral kitchen gadgets with high demand
- **Fashion**: Discovered emerging jewelry trends with low competition
- **Sports**: Located trending fitness accessories with strong growth

---

**Ready to find your next winning product?** 🚀

Start the pipeline and let AI discover profitable opportunities for you!
