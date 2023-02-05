while true
do
    echo $(date): $(vmstat -s -t | grep "used memory") 
    sleep 90
done