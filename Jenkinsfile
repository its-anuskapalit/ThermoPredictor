pipeline {
    agent any

    environment {
        DOCKERHUB_CREDENTIALS = credentials('dockerhub-creds')
        DOCKERHUB_USERNAME = 'anuskap'
        IMAGE_NAME = 'thermalmodel'
    }

    stages {
        stage('Clone') {
            steps {
                // Clone the GitHub repository
                git 'https://github.com/its-anuskapalit/ThermoPredictor.git'
            }
        }

        stage('Build') {
            steps {
                // Install dependencies and prepare the environment
                sh 'pip install --upgrade pip'
                sh 'pip install -r requirements.txt'
            }
        }

        stage('Docker Build & Push') {
            when {
                branch 'main'
            }
            steps {
                script {
                    // Build the Docker image
                    dockerImage = docker.build("${DOCKERHUB_USERNAME}/${IMAGE_NAME}")
                    
                    // Push the Docker image to Docker Hub
                    docker.withRegistry('https://index.docker.io/v1/', DOCKERHUB_CREDENTIALS) {
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                // Use Docker Compose to deploy the application
                sh 'docker-compose down || true'
                sh 'docker-compose up -d'
            }
        }
    }

    post {
        always {
            cleanWs()  // Clean the workspace after the build
        }
    }
}
