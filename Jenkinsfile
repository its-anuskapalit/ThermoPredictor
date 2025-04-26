pipeline {
    agent any

    environment {
        IMAGE_NAME = 'anuskap/thermalmodel'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/its-anuskapalit/ThermoPredictor.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t anuskap/thermalmodel .'
            }
        }

    }

    post {
        always {
            cleanWs()
        }
    }
}
