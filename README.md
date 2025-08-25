# WPF - Winning Product Finder

A powerful product research platform that analyzes eBay, Amazon, Google Trends, and AliExpress data to identify high-potential products for e-commerce businesses.

## 🚀 Live Demo

- **Frontend (GitHub Pages)**: [Your GitHub Pages URL]
- **Backend (Render.com)**: [https://productfinder-k3dy.onrender.com](https://productfinder-k3dy.onrender.com)

## 🏗️ Architecture

This project uses a **frontend/backend separation** approach:

- **Frontend**: Static HTML/CSS/JS hosted on GitHub Pages
- **Backend**: Python Flask API hosted on Render.com
- **Communication**: REST API calls from frontend to backend

## 📁 Project Structure

```
WinningProdAgr/
├── docs/                      # GitHub Pages frontend
│   ├── index.html            # Main landing page
│   └── README.md             # Frontend documentation
├── web_app.py                 # Flask backend API
├── pipeline.py                # Product analysis pipeline
├── etl/                       # Data extraction modules
├── matching.py                # Product matching logic
├── costing.py                 # Cost calculation
├── scoring.py                 # Product scoring
├── config.yaml                # Configuration
├── requirements.txt            # Python dependencies
├── runtime.txt                # Python version for Render
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Deploy Backend to Render.com

1. **Fork/Clone this repository**
2. **Create a Render.com account** at [render.com](https://render.com)
3. **Create a new Web Service**:
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python web_app.py`
   - Environment Variables:
     ```
     EBAY_CLIENT_ID=your_ebay_client_id
     EBAY_CLIENT_SECRET=your_ebay_client_secret
     APIFY_TOKEN=your_apify_token
     KEEPA_KEY=your_keepa_key
     ```

### 2. Deploy Frontend to GitHub Pages

1. **Push the `docs/` folder** to your GitHub repository
2. **Enable GitHub Pages**:
   - Go to Settings → Pages
   - Source: Deploy from a branch
   - Branch: main (or master)
   - Folder: /docs

### 3. Update Backend URL

In your `index.html`, update the backend URL to match your Render deployment:

```html
<!-- Replace this URL with your actual Render URL -->
<a href="https://your-app-name.onrender.com/dashboard" target="_blank">
```

## 🔧 Configuration

### Backend Configuration (`config.yaml`)

```yaml
run_mode: "mvp"
scan:
  min_score: 0.65
  landed_cap_usd: 10.0
  
sources:
  ebay:
    enabled: true
    categories: ["177772", "63514"]
    keywords: ["pet nail grinder", "phone stand"]
    
  trends:
    enabled: true
    keywords: ["pet nail grinder", "phone stand"]
    
  aliexpress:
    enabled: true
    provider: "apify"
    max_results: 20
```

### Environment Variables (`.env`)

```bash
# eBay API
EBAY_CLIENT_ID=your_client_id
EBAY_CLIENT_SECRET=your_client_secret

# Apify (AliExpress search)
APIFY_TOKEN=your_apify_token

# Keepa (Amazon data)
KEEPA_KEY=your_keepa_key
```

## 🎯 Features

- **Multi-Source Data Collection**: eBay, Amazon, Google Trends, AliExpress
- **AI-Powered Matching**: Title and image similarity analysis
- **Cost Analysis**: Landed cost calculation with risk assessment
- **Smart Scoring**: Rule-based product ranking system
- **Real-time Pipeline**: Background processing with progress tracking
- **Dynamic Configuration**: User-configurable search parameters
- **Responsive Web Interface**: Mobile-friendly dashboard

## 🔌 API Endpoints

The backend provides these REST endpoints:

- `GET /api/pipeline_status` - Pipeline status and progress
- `POST /api/start_pipeline` - Start product analysis
- `POST /api/stop_pipeline` - Stop running pipeline
- `GET /api/results_view` - Get all scored products
- `GET /api/results` - Get winning products only

## 🛠️ Development

### Local Development

1. **Install Python 3.12**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```
4. **Run the backend**:
   ```bash
   python web_app.py
   ```
5. **Open frontend**: Open `index_github_pages.html` in your browser

### Testing

```bash
# Test pipeline functionality
python test_simple.py

# Test web app
python web_app.py
```

## 📊 Data Flow

1. **User Configuration**: Frontend sends search parameters to backend
2. **Data Collection**: Backend fetches data from enabled sources
3. **Product Matching**: AI matches market items with suppliers
4. **Cost Analysis**: Calculates landed costs and profit margins
5. **Scoring**: Ranks products by success potential
6. **Results**: Returns scored products to frontend

## 🌐 Deployment

### Render.com Backend

- **Automatic deployment** from GitHub
- **Environment variables** for API keys
- **Free tier** available for testing
- **Custom domain** support

### GitHub Pages Frontend

- **Static hosting** with automatic HTTPS
- **Custom domain** support
- **Automatic deployment** from main branch
- **CDN distribution** worldwide

## 🔒 Security

- **API keys** stored in environment variables
- **CORS** configured for frontend/backend communication
- **Input validation** on all API endpoints
- **Rate limiting** on external API calls

## 📈 Monitoring

- **Pipeline status** tracking
- **Progress indicators** for long-running operations
- **Error logging** with full tracebacks
- **Performance metrics** for optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Create a GitHub issue
- **Documentation**: Check the code comments
- **API Keys**: Contact respective service providers

## 🚀 Next Steps

- [ ] Add Amazon Keepa integration
- [ ] Implement TikTok trend analysis
- [ ] Add product image similarity with CLIP
- [ ] Create Discord bot integration
- [ ] Add export functionality (CSV, Excel)
- [ ] Implement user authentication
- [ ] Add product tracking over time

---

**Built with ❤️ using Python, Flask, and modern web technologies**
