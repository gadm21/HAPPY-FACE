<?php
header('Access-Control-Allow-Origin: *');
$servername = "localhost";
$username = "root";
$password = "root";
$dbname = "vehicledb";

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 
$result ="";

if(!isset($_GET['id'])) {

$sql = "SELECT * FROM v_logs order by created_at DESC limit 0,15";
$result = $conn->query($sql);

} else {

$sql = "SELECT * FROM v_logs WHERE id=" .$_GET['id'] ;
$result = $conn->query($sql);

}



if ($result->num_rows > 0) {
    // output data of each row
    $data =[];

    while($row = mysqli_fetch_assoc($result))
    {
    	$data[] = $row;
      
    }

    echo json_encode(array('data'=>$data));
} else {
    echo "0 results";
}
$conn->close();
?>