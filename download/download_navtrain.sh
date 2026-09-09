for split in {2..4}; do
    echo "Extracting file navtrain_current_${split}.tgz"
    tar -xzf navtrain_current_${split}.tgz
    rm navtrain_current_${split}.tgz

    rsync -rv current_split_${split}/* trainval_sensor_blobs/trainval
    rm -r current_split_${split}
done

for split in {1..4}; do
    echo "Extracting file navtrain_history_${split}.tgz"
    tar -xzf navtrain_history_${split}.tgz
    rm navtrain_history_${split}.tgz

    rsync -rv history_split_${split}/* trainval_sensor_blobs/trainval
    rm -r history_split_${split}
done
