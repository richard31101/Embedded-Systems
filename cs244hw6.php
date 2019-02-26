<?php
echo "begin to write CSV file.";
/*
$json = '{
          "data": [
                [29823,21826,-0.017014,-0.110089,-0.97279],
                [29963,21845,-0.019015,-0.115094,-0.963783],
                [30003,21851,-0.009007,-0.105085,-0.9848]
          ],
          "activity": Sitting
        }';
*/

$dataDecode = json_decode($_POST['data'],true);

if(file_exists ( "team20_assignment6_(".$dataDecode['direction'].").csv" ))
    $file = fopen("team20_assignment6_(".$dataDecode['direction'].").csv","a");
else
    $file = fopen("team20_assignment6_(".$dataDecode['direction'].").csv","w");

foreach ($dataDecode['data'] as $list)
{
    fputcsv($file,$list);
}

fclose($file);

echo "write CSV file complete.";

?>
