#/bin/sh +ex

mkdir -p results/usher

# These are the available files:
# public-latest.version.txt
# public-latest.metadata.tsv.gz
# public-latest.all.masked.pb.gz
# public-latest.all.masked.vcf.gz
# public-latest.all.nwk.gz
url='http://hgdownload.soe.ucsc.edu/goldenPath/wuhCor1/UShER_SARS-CoV-2'
for name in version.txt metadata.tsv.gz all.masked.pb.gz
do
  curl $url/public-latest.$name -o results/usher/$name
done

gunzip -kvf results/usher/*.gz
