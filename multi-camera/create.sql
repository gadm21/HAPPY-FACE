CREATE DATABASE face;

CREATE TABLE Demographic (
    ID int NOT NULL AUTO_INCREMENT,
    Gender char(1),
    GenderConfidence float(2),
    AgeLow int,
    AgeHigh int,
    Date TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
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