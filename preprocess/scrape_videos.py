from pytube import YouTube
import urllib.request, json
import time
import os


def get_category(cat_id, num_to_get, path=None):
	CATEGORY_BASE_URL = "https://storage.googleapis.com/data.yt8m.org/1/j/"
	id_list = []
	print("Getting category", cat_id)
	with urllib.request.urlopen(CATEGORY_BASE_URL + cat_id + ".js") as url:
	    data = url.read().decode()
	    id_list = data.split('","')[2:-1]

	print("Got category...")
	num_downloaded = 0
	i = 0
	while num_downloaded < num_to_get:
		vid_id = id_list[i]
		result = "failure"
		try:

			result = get_video(vid_id, tag=cat_id+"_"+str(num_downloaded), path=path)
		except Exception as e:
			print("Failure:", vid_id, cat_id, i) 
			print(e)

		if (result == "success"):
			num_downloaded += 1

		print(num_downloaded, "download,", i, "total")
		i+=1



def get_video(vid_id, resolution="720p", tag="", path=None):
	YT_BASE_URL = "https://www.youtube.com/watch?v="
	
	yt = YouTube(YT_BASE_URL + vid_id)
	stream = yt.streams.filter(res=resolution).first()

	if (stream):
		name = vid_id + "_" + tag
		stream.download(output_path=path, filename=name)
		time.sleep(5)
		return "success"
	else:
		return "failure"

if __name__ == "__main__":
	category_ids = {
		"vehicle":"07yv9",
		"concert":"01jddz",
		"dance":"026bk",
		"football":"02vx4",
		"animal":"0jbk",
		"food":"02wbm",
		"racing":"0dfbw",
		"outdoor_recreation":"05b0n7k",
		"fashion":"032tl",
		"call_of_duty":"026wy8d",
		"road":"06gfj",
		"race_track":"01r_pn",
		"basketball":"018w8",
		"train":"07jdr",
		"driving":"0kw6d",
		"dog":"0bt9lr",
		"mobile_phone":"050k8",
		"american_football":"0jm_",
		"athlete":"01445t",
		"combat":"0byj4",
		"sports_game":"022dc6",
		"dashcam":"0r4kr10",
		"talent_show":"0byb_x",
		"festival":"0hr5k",
		"weight_training":"0c4f_",
		"nightclub":"01sg9l",
		"tree":"07j7r",
		"model_aircraft":"0l8fl",
		"running":"06h7j",
		"train_station":"0py27",
		"cat":"01yrx",
		"home_improvement":"03n2_q",
		"highway":"0cz_0",
		"eating":"01f5gx",
		"ocean":"05kq4",
		"hotel":"03pty",
		"helicopter":"09ct_",
		"room":"06ht1",
		"figure_skating":"02_5h",
		"robot":"06fgw",
		"roller_coaster":"010l12",
		"scooter":"0dmq2",
		"sketch_comedy":"0dm00",
		"jumping":"0by3w",
		"need_for_speed":"02ks74",
		"resort":"02dkrm",
		"pickup_truck":"0cvq3",
		"underwater_diving":"09kjpq",
		"camping":"01h6d4",
		"livestock":"0ch8v",
		"teacher":"01d30f",
		"city":"01n32",
		"family":"09dhj",
		"cricket":"09xp_",
		"costume":"0250x",
		"snowboard":"06__v",
		"zoo":"089v3",
		"parachuting":"05zmb",
		"night":"01d74z",
		"walt_disney_world":"09b1k"
	}
	
	path = "videos"
	categories = sorted(category_ids.keys())
	for category in categories:
		# directory = os.path.join(path, category)
		directory = path
		if not os.path.exists(directory):
		    os.makedirs(directory)

		get_category(category_ids[category], 5, directory)
		print("Finished category", category + "...")
