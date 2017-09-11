import scrapy
import json
import io
import os
import os.path

current_dir = os.path.dirname(__file__)
comments_path = os.path.join(current_dir,"../../../MoviesComments/movies_comments.json")

path = comments_path
if os.path.exists(path):
    os.remove(path)

    

class MoviesSpider(scrapy.Spider):
    name = "movies_comments"
    def start_requests(self):
        urls = [
            'http://www.fandango.com/moviesintheaters',
        ]
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    
            

    def get_comments(self, response):
        comments = response.meta['comments']
        movieTitle = response.meta['title']
        for comment in response.css('div.movie-review-item-content::text').extract():
            comment = comment.replace('\r\n', ' ')
            comment = comment.replace('\n\r', ' ')
            comment = comment.replace('\n',' ')
            comment = comment.replace('\"', '')
            comment = comment.strip()
            comments.append(comment)

# uncomment to search the rest of comments pages
# Crawling will take 30 - 40 minutes
# JSON file will be 2+GB

##        next_pages = response.css('span.showtimes-pagination-numbers a::attr(href)').extract()
##        if next_pages is not None:
##            for next_page in next_pages:
##                next_page = response.urljoin(next_page)
##                request = scrapy.Request(next_page, callback=self.get_comments)
##                request.meta['comments'] = comments
##                request.meta['title'] = movieTitle
##                yield request

        output_file = open(path, 'a', encoding='utf-8')
        json.dump({'title': movieTitle, 'comments': comments}, output_file, ensure_ascii=False)
        
        output_file.write('\n')
        
        output_file.close()
        
    def parse(self, response):
        
        movies = response.xpath("//li[@class='visual-item']//div[@class='visual-detail']/a")
        for movie in movies:
            movieTitle = movie.xpath('text()').extract_first().strip()
            print(movieTitle)
            
            movieComments = []
            
            movieUrl = movie.xpath('@href').extract_first()
            movieUrl = movieUrl.replace('movieoverview', 'moviereviews')
            print(movieUrl)
            
            movie_page = response.urljoin(movieUrl)
            request = scrapy.Request(movie_page, callback=self.get_comments)
            request.meta['comments'] = movieComments
            request.meta['title'] = movieTitle
            yield request
            

