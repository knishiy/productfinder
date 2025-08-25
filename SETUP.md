# 🚀 Winning Product Pipeline - Setup Guide

## ✅ **What's Working Now:**

- ✅ Python 3.12 compatibility
- ✅ All required packages installed
- ✅ Pipeline structure and error handling
- ✅ Configuration system
- ✅ Component initialization
- ✅ Graceful error handling

## 🔑 **Required API Keys:**

### 1. **eBay API Keys**
1. Go to [eBay Developers](https://developer.ebay.com/)
2. Create a new application
3. Get your **Client ID** and **Client Secret**
4. Add to `.env` file

### 2. **Apify Token (for AliExpress)**
1. Go to [Apify](https://apify.com/)
2. Create an account
3. Get your **API Token**
4. Add to `.env` file

## 📝 **Setup Steps:**

### Step 1: Create `.env` file
Copy `env_template.txt` to `.env` and fill in your actual API keys:

```bash
# Copy the template
copy env_template.txt .env

# Edit .env with your actual keys
notepad .env
```

Your `.env` should look like:
```env
# eBay API (prod keys)
EBAY_CLIENT_ID=your_actual_ebay_client_id_here
EBAY_CLIENT_SECRET=your_actual_ebay_client_secret_here

# Apify (temporary AliExpress search)
APIFY_TOKEN=apify_your_actual_apify_token_here

# Optional later
KEEPA_KEY=your_keepa_key_here
```

### Step 2: Test the setup
```bash
py -3.12 test_simple.py
```

All tests should pass once you have valid API keys.

### Step 3: Run the web interface
```bash
py -3.12 web_app.py
```

Open http://localhost:5000 in your browser.

## 🎯 **What the Pipeline Does:**

1. **eBay Data Collection**: Searches for products in configured categories
2. **Google Trends Analysis**: Calculates 14-day growth for keywords
3. **AliExpress Supplier Search**: Finds suppliers via Apify
4. **Product Matching**: Matches market products with suppliers
5. **Scoring & Ranking**: Calculates winning product scores
6. **Results Display**: Shows results in web dashboard

## 🔧 **Configuration Options:**

Edit `config.yaml` to customize:
- **Categories**: eBay category IDs to search
- **Keywords**: Specific product keywords to monitor
- **Scoring weights**: Adjust importance of different factors
- **Risk thresholds**: Modify acceptable risk levels

## 🚨 **Troubleshooting:**

### Common Issues:

1. **"API keys not configured"**
   - Check your `.env` file exists
   - Verify API keys are correct
   - Restart the application after changing `.env`

2. **"Module import failed"**
   - Ensure you're using Python 3.12
   - Run `py -3.12 -m pip install -r requirements.txt`

3. **"Pipeline failed"**
   - Check the logs in `pipeline.log`
   - Verify API keys are valid
   - Check internet connection

### Getting Help:

1. Check the logs: `pipeline.log`
2. Run tests: `py -3.12 test_simple.py`
3. Check configuration: `config.yaml`

## 🎉 **Success Indicators:**

- ✅ All tests pass in `test_simple.py`
- ✅ Web interface loads at http://localhost:5000
- ✅ Pipeline starts without errors
- ✅ Data collection begins
- ✅ Results appear in dashboard

## 🔮 **Next Steps After Setup:**

1. **Customize Categories**: Add your target eBay categories
2. **Adjust Keywords**: Focus on specific product types
3. **Fine-tune Scoring**: Adjust weights for your business
4. **Monitor Results**: Check pipeline performance
5. **Scale Up**: Add more data sources later

---

**Ready to find winning products?** 🚀

Set up your API keys and start the pipeline!
