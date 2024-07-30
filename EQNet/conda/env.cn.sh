cd conda/
cp $1 $2
sed -i -e s,github.com/JonathonLuiten/TrackEval.git,gitee.com/hyuyao/TrackEval_Clone_240331, $2
sed -i -e s,github.com/argoverse/av2-api.git,gitee.com/hyuyao/av2-api_clone240331, $2
