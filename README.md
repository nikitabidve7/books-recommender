# Book Recommender System

A machine learning-based Book Recommender System developed using Python and Streamlit. The application recommends books based on user selection and is containerized using Docker and deployed on AWS EC2.

##  Live Demo

**Application URL:**

http://bookrecommender.duckdns.org:8501

---

##  Project Overview

The Book Recommender System helps users discover books similar to their interests. The recommendation engine analyzes book information and suggests relevant books based on the selected title.

This project demonstrates:

* Recommendation Systems
* Machine Learning
* Streamlit Web Application Development
* Docker Containerization
* Cloud Deployment using AWS EC2
* DNS Configuration using Duck DNS

---

##  Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* Scikit-learn
* Streamlit

### Deployment & Cloud

* Docker
* Docker Hub
* AWS EC2 (Ubuntu Linux)
* Elastic IP
* Duck DNS

---

Project Structure
Book-Recommender/
│
├── app.py
├── bookrec.csv
├── requirements.txt
├── Dockerfile
└── README.md


---


## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t book-recommender .
```

### Run Docker Container

```bash
docker run -p 8501:8501 book-recommender
```

Access the application:

```text
http://localhost:8501
```

---

## ☁️ AWS Deployment

The application is deployed on AWS EC2 using Docker.

### Deployment Steps

1. Create AWS EC2 Ubuntu Instance
2. Configure Security Groups
3. Install Docker
4. Push Docker Image to Docker Hub
5. Pull Docker Image on EC2
6. Run Docker Container
7. Configure Elastic IP
8. Configure Duck DNS

### Docker Hub Image

```bash
docker pull nikitabidve/book-recommender:v1
```

---

## 🎯 Features

* User-friendly interface
* Book recommendation engine
* Fast response time
* Cloud deployment
* Dockerized application
* Publicly accessible through DNS

---

## 📈 Future Enhancements

* Content-based recommendation system
* Collaborative filtering
* User authentication
* Personalized recommendations
* Book cover visualization
* Database integration
* HTTPS support
* CI/CD Pipeline

---

##  Developer

**Nikita Bidve**

B.Tech CSE (AIML)

LinkedIn:
http://www.linkedin.com/in/nikita-bidve-856450280

Email:
nikitabidve7@gmail.com

---

## 📄 License

This project is developed for educational and learning purposes.
