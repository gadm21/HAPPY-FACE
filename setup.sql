CREATE DATABASE face;

USE face;

CREATE TABLE FaceExitTime (
    ID int NOT NULL AUTO_INCREMENT,
    ExitedTime DATETIME,
    PRIMARY KEY (ID)
);

CREATE TABLE Demographic (
    ID int NOT NULL AUTO_INCREMENT,
    Gender char(1),
    GenderConfidence float(2),
    Emotion char(20),
    AgeLow int,
    AgeHigh int,
    DetectedTime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ExitedTime DATETIME,
    PRIMARY KEY (ID)
);

CREATE TABLE Image(
    ID int NOT NULL,
    ImageBlob blob NOT NULL
);

CREATE Table Info(
    ID int NOT NULL,
    Location varchar(255) NOT NULL
);

/*
INSERT INTO Demographic (gender,AgeLow,AgeHigh) 
VALUES ('M',10,20);

SELECT LAST_INSERT_ID();

INSERT INTO recognition (name,address) 
VALUES ('oka','add');

DROP TABLE Demographic;

DELETE FROM Demographic;
DELETE FROM Info;
*/
