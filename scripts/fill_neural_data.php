<?php
$servername = "localhost";
$username = "apache_bip_user";
$password = "b1p1$@w3s0m3";
$dbname = "pmc_papers";

function pick5Categories($score){
	if($score < 20)
		return 5;
	else if($score < 92)
		return 4; 
	else if ($score < 386)
		return 3;
	else if($score < 1420)
		return 2;
	else 
		return 1;
}

function pick7Categories($score){
	if($score < 11)
		return 7;
	else if($score < 38)
		return 6; 
	else if ($score < 107)
		return 5;
	else if($score < 290)
		return 4;
	else if($score < 773)
		return 3;
	else if($score < 1976)
		return 2;
	else
		return 1;
}

// Create connection
$conn = new mysqli($servername, $username, $password, $dbname);

// Check connection
if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
} 

$handle = fopen("pmc_optimal_fcc_2013_2016.txt", "r");
while (($line = fgets($handle)) !== false) {
	$line = trim(preg_replace('/\s\s+/', ' ', $line));
    $tokens = explode("\t", $line);
    $score = intval($tokens[1]);
    $sql = "SELECT pmc, year, journal, " . $score . " as score, " . pick5Categories($score) . 
    " as category5, " . pick7Categories($score) ." as category7, authors FROM pmc_paper WHERE pmc = '" . $tokens[0] . "'";
    // print_r($sql);
	$result = $conn->query($sql);
	$row = $result->fetch_assoc();
	echo implode("\t", $row) . "\n";
    
}

fclose($handle);
$conn->close();