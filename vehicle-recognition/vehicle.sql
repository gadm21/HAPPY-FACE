CREATE DATABASE IF NOT EXISTS `vehicledb`;
USE `vehicledb` ;

DROP TABLE IF EXISTS `v_logs`;
/*!40101 SET @saved_cs_client     = @@character_set_client */;
/*!40101 SET character_set_client = utf8 */;
CREATE TABLE `v_logs` (
  `id` int NOT NULL AUTO_INCREMENT,
  `mac` varchar(256) CHARACTER SET utf8 NOT NULL COMMENT 'mac',
  `plate_id` varchar(256) CHARACTER SET utf8 NOT NULL,
  `frame_img_path` varchar(256) CHARACTER SET utf8 NOT NULL,
  `timestamp` varchar(255) CHARACTER SET utf8 NOT NULL,
  `updated_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'last operation date',
  `created_at` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `id` (`id`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;