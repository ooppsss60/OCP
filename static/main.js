$(function () {

    var colors = ['green', 'purple', 'orange', 'red', 'blue', 'grey']
    var classes = ['good', 'color', 'cut', 'hole', 'thread', 'metal']
    for (let i = 0; i < 6; i++) {
        let label = $('<li/>')
        label.addClass('list-inline-item')
        label.append(classes[i])
        label.css('color', colors[i])
        label.css('font-size', '31px')
        $('#classes').append(label);
    }

    $.get("json", function (json) {
        console.log(json)
        for (let url in json) {
            let color = colors[json[url]['pred']];
            let probability = $('<ul>',  {'class': 'list-unstyled'});
            for (let probClass in json[url]['prob']){
                let prob = Math.round(json[url]['prob'][probClass] * 1000) / 10
                probability.append('<li>'+classes[probClass]+': '+prob+'%</li>')
            }
            let image = $('<img/>', {
                'class': 'img-thumbnail',
                'src': url,
                'data-toggle': "popover",
                'data-trigger': "hover",
                'data-html': "true",
                'data-content': probability[0].outerHTML
            });
            image.addClass('animated');
            image.addClass('zoomIn');
            image.css('border', '3px solid ' + color);
            $('#image_col').append(image);
        }
        $('[data-toggle="popover"]').popover()
    });
})


