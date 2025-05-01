CREATE DATABASE attendance_system;
USE attendance_system;

CREATE TABLE users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(50) UNIQUE,
    name VARCHAR(100)
);

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id VARCHAR(50),
    timestamp DATETIME,
    latitude FLOAT,
    longitude FLOAT
);
