# Streamlit for Machine Learning Model Serving

## 🌟 Learning Objectives

By completing this exercise, you will learn:

1. **Why Streamlit Matters**: Understand how Streamlit revolutionizes ML app development:
   - Rapid prototyping (build apps in minutes, not days)
   - No frontend knowledge required (Python-only)
   - Interactive widgets built for ML workflows
   - Automatic reactive updates

2. **Streamlit vs Traditional Web Development**: See the difference:
   - **Traditional**: Flask/Django + React/Vue + HTML/CSS/JavaScript (weeks of work)
   - **Streamlit**: Pure Python, interactive UI in hours

3. **Production ML Interfaces**: Build user-friendly interfaces for ML models that:
   - Accept user inputs through intuitive forms
   - Display predictions with visualizations
   - Support batch predictions
   - Track prediction history

## 📚 What You'll Build

A complete interactive machine learning application that:
- Serves a diabetes progression prediction model
- Provides an intuitive web interface for single predictions
- Supports batch predictions via CSV upload
- Visualizes results with charts and gauges
- Tracks prediction history
- Displays model information and feature importance

## 🎯 The Problem We're Solving

### Traditional Approach (Flask + React)

**The Challenge:**
- Need to learn HTML, CSS, JavaScript
- Separate frontend and backend codebases
- Complex state management
- Weeks to build a simple ML interface
- Requires full-stack development skills

**The Solution:**
- Streamlit: Pure Python, no frontend knowledge needed
- Built-in widgets for ML workflows
- Automatic reactive updates
- Deploy in hours, not weeks

## 🚀 Quick Start

### Option 1: Local Development

1. **Install dependencies:**
   ```bash
   poetry install
   ```

2. **Train the model (if not already done):**
   ```bash
   poetry run jupyter-lab
   # Open and run Train_ML_diabetes.ipynb
   ```

3. **Run the Streamlit app:**
   ```bash
   poetry run streamlit run app.py
   ```

4. **Open in browser:**
   The app will automatically open at http://localhost:8501

### Option 2: Docker

1. **Build the Docker image:**
   ```bash
   docker build -t diabetes-streamlit-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 8501:8501 diabetes-streamlit-app
   ```

3. **Access the app:**
   Open http://localhost:8501 in your browser

### Option 3: Quick Start Script

```bash
./quick_start.sh
```

## 📖 Step-by-Step Guide

For detailed instructions, see [STUDENT_GUIDE.md](STUDENT_GUIDE.md)

## 🏗️ Project Structure

```
diabetes-streamlit/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration management
├── best_diabetes_model.pkl    # Trained model (generate via notebook)
├── Train_ML_diabetes.ipynb    # Model training notebook
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── Dockerfile                  # Container setup
└── README.md                  # This file
```

## 🔍 Understanding Streamlit

### What is Streamlit?

Streamlit is an open-source Python framework for building interactive web applications for machine learning and data science.

### Key Benefits

1. **Rapid Prototyping**
   - Build ML apps in minutes, not days
   - Iterate quickly with instant updates
   - Perfect for demos and prototypes

2. **No Frontend Knowledge Required**
   - Pure Python - no HTML/CSS/JavaScript
   - Built-in widgets for common ML tasks
   - Automatic layout and styling

3. **Interactive by Default**
   - Widgets trigger automatic reruns
   - Reactive programming model
   - Real-time updates

4. **Built-in Visualization**
   - Integration with Plotly, Altair, Matplotlib
   - Charts, graphs, and visualizations out of the box
   - No need for separate visualization libraries

5. **Easy Deployment**
   - Deploy to Streamlit Cloud with one click
   - Docker support for custom deployments
   - Share apps via URL

### Streamlit vs Traditional Web Apps

| Feature | Traditional (Flask/React) | Streamlit |
|---------|-------------------------|-----------|
| **Setup Time** | Days to weeks | Hours |
| **Languages** | Python + HTML/CSS/JS | Python only |
| **Codebase** | Frontend + Backend | Single file |
| **Learning Curve** | Steep (full-stack) | Gentle (Python) |
| **Best For** | Complex web apps | ML/data apps |

## 📊 The Diabetes Dataset

The diabetes dataset contains:
- **10 Features**: Age, Sex, BMI, BP, and 6 blood serum measurements (S1-S6)
- **Target**: Diabetes progression score (continuous value)
- **Model**: Ridge Regression (regularized linear regression)

### Feature Descriptions

- **Age**: Age in years
- **Sex**: Gender (0 = female, 1 = male)
- **BMI**: Body Mass Index (kg/m²)
- **BP**: Average blood pressure (mm Hg)
- **S1**: Total cholesterol (mg/dl)
- **S2**: Low-density lipoproteins (mg/dl)
- **S3**: High-density lipoproteins (mg/dl)
- **S4**: Total cholesterol / HDL ratio
- **S5**: Log of serum triglycerides level
- **S6**: Blood sugar level (mg/dl)

## 🎨 Application Features

### 1. Prediction Page
- Interactive input form with sliders
- Quick preset buttons (Low/Medium/High risk)
- Real-time prediction with gauge visualization
- Feature descriptions and validation

### 2. Model Info Page
- Model performance metrics (R², MAPE)
- Feature importance visualization
- Dataset information
- Model details

### 3. Prediction History
- Track all predictions made
- Visualize prediction trends
- Download history as CSV
- Statistics and summaries

### 4. Batch Prediction
- Upload CSV file with multiple patients
- Bulk predictions
- Download results
- Template download

## 🔧 Configuration

Configuration is managed through:
- `config.py`: Feature definitions, model path, defaults
- `.streamlit/config.toml`: Streamlit theme and server settings
- Environment variables: Override defaults if needed

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t diabetes-streamlit-app .
```

### Run Container
```bash
docker run -p 8501:8501 diabetes-streamlit-app
```

### With Environment Variables
```bash
docker run -p 8501:8501 \
  -e MODEL_PATH=/app/model.pkl \
  diabetes-streamlit-app
```

## 🧪 Testing

Test the application by:
1. Making single predictions with different inputs
2. Using preset profiles
3. Uploading a CSV for batch predictions
4. Checking prediction history

## 🐛 Troubleshooting

### Model file not found
- Run the training notebook first: `Train_ML_diabetes.ipynb`
- Ensure `best_diabetes_model.pkl` is in the project root

### Streamlit won't start
- Check if port 8501 is already in use
- Verify dependencies are installed: `poetry install`

### Import errors
- Ensure you're in the correct directory
- Check Python version (3.10+)
- Reinstall dependencies: `poetry install --no-cache`

### Docker build fails
- Ensure Docker is running
- Check disk space: `docker system df`
- Try cleaning: `docker system prune`

## 🎓 Next Steps

After completing this exercise:

1. **Customize**: Modify the UI, add more visualizations
2. **Extend**: Add more models, comparison features
3. **Deploy**: Deploy to Streamlit Cloud or your own server
4. **Integrate**: Connect to databases, APIs, or other services
5. **Learn More**: Explore Streamlit's advanced features (caching, session state, multi-page apps)

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Community](https://discuss.streamlit.io/)
- [Deployment Guide](https://docs.streamlit.io/streamlit-community-cloud)

## 👨‍🏫 For Instructors

See [INSTRUCTOR_NOTES.md](INSTRUCTOR_NOTES.md) for teaching guide and concept explanations.

## 📝 License

This is an educational exercise for teaching Streamlit and ML model serving.





