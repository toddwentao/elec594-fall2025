DATASET=/workspace/capture/DSC_0010

colmap feature_extractor \
  --database_path $DATASET/database.db \
  --image_path $DATASET/images \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model SIMPLE_RADIAL \
  --SiftExtraction.use_gpu 1

colmap sequential_matcher --database_path $DATASET/database.db --SiftMatching.use_gpu 1

mkdir -p $DATASET/sparse
colmap mapper \
  --database_path $DATASET/database.db \
  --image_path $DATASET/images \
  --output_path $DATASET/sparse \
  --Mapper.ba_refine_focal_length 0 \
  --Mapper.ba_refine_principal_point 0 \
  --Mapper.tri_min_angle 2

# STOP!
# check how many pairs of models are registered
ls -l $DATASET/sparse/0

##################################

for i in 0 1; do
  if [ -d "$DATASET/sparse/$i" ]; then
    echo "== Model $i =="
    colmap model_analyzer --path $DATASET/sparse/$i | sed -n '1,25p'
  fi
done

##################################

mkdir -p $DATASET/sparse_refined
colmap bundle_adjuster \
  --input_path  $DATASET/sparse/0 \
  --output_path $DATASET/sparse_refined \
  --BundleAdjustment.refine_focal_length 1 \
  --BundleAdjustment.refine_principal_point 0 \
  --BundleAdjustment.refine_extra_params 1

############
colmap model_converter \
  --input_path  $DATASET/sparse_refined \
  --output_path $DATASET/sparse_refined_txt \
  --output_type TXT

