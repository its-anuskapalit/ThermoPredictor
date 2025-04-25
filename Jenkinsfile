pipeline {
    agent any

    environment {
        IMAGE_NAME = 'anuskap/thermalmodel'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/its-anuskapalit/ThermoPredictor.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${IMAGE_NAME}")
                }
            }
        }

        stage('Deploy with Docker Compose') {
            steps {
                script {
                    sh 'docker-compose down || true'
                    sh 'docker-compose up -d'
                }
            }
        }
    }

    post {
        always {
            cleanWs()
        }
    }
}
