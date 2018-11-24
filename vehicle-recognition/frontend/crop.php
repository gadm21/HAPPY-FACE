<?php
ini_set('display_errors', 1);
ini_set('display_startup_errors', 1);
error_reporting(E_ALL);

header('Content-Type: image/png');

$image = $_GET['src']; // the image to crop
$dest_image = 'cropped/' . $_GET['img_name']; // make sure the directory is writeable

$fh = fopen($dest_image, 'w') or die("Can't create file");


$img = imagecreatetruecolor('200','150');
$org_img = LoadPNG($image);
$ims = getimagesize($image);

$image_src_name = str_replace('.png', '', $_GET['src_img_name']);
$image_src_name_1 = explode('_', $image_src_name);
$image_src_name_2 = explode(':', $image_src_name_1[3]);


imagecopy($img,$org_img, 0, 0, $image_src_name_2[0], $image_src_name_2[1], $image_src_name_2[2], $image_src_name_2[3]);
imagepng($img,$dest_image,9);
imagedestroy($img);
echo json_encode(array('data'=>$dest_image));



function LoadPNG($imgname)
{
    /* Attempt to open */
    $im = @imagecreatefrompng($imgname);

    /* See if it failed */
    if(!$im)
    {
        /* Create a blank image */
        $im  = imagecreatetruecolor(150, 30);
        $bgc = imagecolorallocate($im, 255, 255, 255);
        $tc  = imagecolorallocate($im, 0, 0, 0);

        imagefilledrectangle($im, 0, 0, 150, 30, $bgc);

        /* Output an error message */
        imagestring($im, 1, 5, 5, 'Error loading ' . $imgname, $tc);
    }

    return $im;
}


